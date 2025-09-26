from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import yaml
import optuna
from optuna.samplers import GPSampler
import torch
import gc
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

class Gemma3InstructTrainer():
    def __init__(self, config):
        self.default_config = config
        self.dataset = None
        self.tokenizer = None
        self._data_loaded = False

    def _preprocess_data(self, batch):
        """Preprocess data using tokenizer chat template"""
        if self.tokenizer is None:
            # Return unchanged if tokenizer not available yet
            return batch
            
        texts = [
            self.tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            ) for convo in batch["messages"]
        ]
        return {'text': texts}

    def _load_data(self):
        """Load datasets with caching"""
        if self._data_loaded and self.dataset is not None:
            print("Data already loaded, skipping...")
            return
            
        print("Loading datasets from files (checking cache first)...")
        data_files = {
            'train': str(self.default_config['dataset']['train_path']),
            'valid': str(self.default_config['dataset']['valid_path']),
            'test': str(self.default_config['dataset']['test_path']),
        }

        self.dataset = load_dataset(
            'json', 
            data_files=data_files,
            cache_dir="./dataset_cache"
        )
        
        self._data_loaded = True
        print("Data loading completed.")

    def _apply_preprocessing(self):
        """Apply preprocessing after tokenizer is loaded"""
        if self.tokenizer is None:
            print("Warning: Tokenizer not loaded, skipping preprocessing")
            return
        
        self.dataset['train'] = self.dataset['train'].map(
            self._preprocess_data, 
            batched=True,
            desc="Processing train data"
        )
        self.dataset['valid'] = self.dataset['valid'].map(
            self._preprocess_data, 
            batched=True,
            desc="Processing valid data"
        )
        self.dataset['test'] = self.dataset['test'].map(
            self._preprocess_data, 
            batched=True,
            desc="Processing test data"
        )

    def _create_hp_config(self, trial, base_config):
        """Create hyperparameter configuration for trial"""
        hp_space = self.default_config.get('hp_search', {}).get('hp_space', {})
        
        # Create a copy of base config
        trial_config = base_config.copy()
        model_config = trial_config.get('model', {}).copy()
        peft_config = trial_config.get('peft_config', {}).copy()
        
        # Sample hyperparameters based on hp_space configuration
        for param, space_config in hp_space.items():
            if param in ['learning_rate', 'weight_decay', 'lora_dropout']:
                if space_config['type'] == 'loguniform':
                    low = float(space_config['low'])
                    high = float(space_config['high'])
                    value = trial.suggest_float(param, low, high, log=True)
                elif space_config['type'] == 'uniform':
                    low = float(space_config['low'])
                    high = float(space_config['high'])
                    value = trial.suggest_float(param, low, high)
            elif param in ['r', 'lora_alpha', 'per_device_train_batch_size', 'gradient_accumulation_steps', 'warmup_steps', 'num_train_epochs']:
                low = int(space_config['low'])
                high = int(space_config['high'])
                value = trial.suggest_int(param, low, high)
            elif param in ['optim', 'lr_scheduler_type']:
                choices = list(space_config['choices'])
                value = trial.suggest_categorical(param, choices)
            else:
                continue
            
            # Assign to appropriate config section
            if param in ['r', 'lora_alpha', 'lora_dropout']:
                peft_config[param] = value
            else:
                model_config[param] = value
        
        trial_config['model'] = model_config
        trial_config['peft_config'] = peft_config
        
        return trial_config

    def _train_model(self, config, trial=None, skip_data_loading=False):
        """Train model with given configuration"""
        model_config = config.get('model', {})
        peft_config = config.get('peft_config', {})

        # Load data if not already loaded
        if not skip_data_loading and self.dataset is None:
            self._load_data()

        # Load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(model_config.get('model_name')),
            max_seq_length=int(model_config.get('max_length')),
            load_in_4bit=bool(model_config.get('load_in_4bit')),
            load_in_8bit=bool(model_config.get('load_in_8bit')),
            torch_dtype=model_config.get('torch_dtype'), 
        )
        
        if self.tokenizer is None:
            self.tokenizer = tokenizer
            self._apply_preprocessing()

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(peft_config.get('r')), 
            finetune_mlp_modules=bool(peft_config.get('finetune_mlp_modules')),  
            target_modules=list(peft_config.get('target_modules')), 
            lora_alpha=int(peft_config.get('lora_alpha')),  
            lora_dropout=float(peft_config.get('lora_dropout')),  
            bias=peft_config.get('bias'), 
            random_state=int(peft_config.get('random_state')) if peft_config.get('random_state') is not None else None, 
            use_rslora=bool(peft_config.get('use_rslora')),  
        )

        args = SFTConfig(
            dataset_text_field=str("text"),
            output_dir=str(model_config.get("output_dir", "results")),
            per_device_train_batch_size=int(model_config.get("per_device_train_batch_size")),
            gradient_accumulation_steps=int(model_config.get("gradient_accumulation_steps")),
            warmup_steps=int(model_config.get("warmup_steps")),
            num_train_epochs=int(model_config.get("num_train_epochs")),
            max_steps=int(model_config.get("max_steps")),
            learning_rate=float(model_config.get("learning_rate")),
            logging_steps=int(model_config.get("logging_steps")),
            optim=str(model_config.get("optim")),
            weight_decay=float(model_config.get("weight_decay")),
            lr_scheduler_type=str(model_config.get("lr_scheduler_type")),
            seed=int(model_config.get("seed")),
            report_to=str(model_config.get("report_to")),
            save_strategy=str(model_config.get("save_strategy", "epoch")),
            eval_strategy=str(model_config.get("eval_strategy", "epoch")),
            gradient_checkpointing=bool(model_config.get("gradient_checkpointing", False)),
            packing=bool(model_config.get("packing", False)),
            fp16=bool(model_config.get("fp16", True)),
            bf16=bool(model_config.get("bf16", False)),
            device_map={'':torch.cuda.current_device()}
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['valid'],
            args=args
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

        train_results = trainer.train()
        eval_results = trainer.evaluate()
        
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        if trial:
            return eval_results.get('eval_loss', float('inf'))
        else:
            return {
                'train_results': train_results,
                'eval_results': eval_results
            }

    def _run_trial(self, config):
        hp_search_config = self.default_config.get('hp_search', {})
        n_trials = int(hp_search_config.get('n_trials', 6))
        
        if self.dataset is None:
            print("Loading data for hyperparameter search...")
            self._load_data()

        study = optuna.create_study(
            direction='minimize',
            sampler=GPSampler(),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5)
        )
        
        def objective(trial):
            trial_config = self._create_hp_config(trial, config)
            return self._train_model(trial_config, trial=trial, skip_data_loading=True)
        
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                catch=(Exception, )
            )
            print(f"Best parameters: {study.best_params}")
            print(f"Best value: {study.best_value}")
            
            return study.best_params
            
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            if study.best_params:
                return study.best_params
            else:
                return None

    def train(self):
        """Main training method"""
        if self.default_config.get('hp_search', {}).get('enable'):
            print("Starting hyperparameter search...")
            best_parameters = self._run_trial(self.default_config)
            
            if best_parameters:
                print("Training final model with best parameters...")
                final_config = self.default_config.copy()
                
                for param, value in best_parameters.items():
                    if param in ['r', 'lora_alpha', 'lora_dropout']:
                        final_config['peft_config'][param] = value
                    else:
                        final_config['model'][param] = value
                
                # Data already loaded, skip loading
                return self._train_model(final_config, skip_data_loading=True)
            else:
                print("No best parameters found, training with default config...")
                # Data already loaded during HP search, skip loading
                return self._train_model(self.default_config, skip_data_loading=True)
        else:
            return self._train_model(self.default_config)