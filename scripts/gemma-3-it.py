import yaml
import optuna
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from datasets import load_dataset
import gc

class Gemma3Finetuner():
    def __init__(self, config):
        self.default_config = config

    def _formatting_func(self, dataset):
        texts = []
        for messages in dataset["messages"]:
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(formatted_text)
        return {"text": texts}

    def _load_data(self):
        data_files = {
            'train' : self.default_config['dataset']['train_path'],
            'valid' : self.default_config['dataset']['valid_path'],
            'test'  : self.default_config['dataset']['test_path'],
        }

        self.dataset = load_dataset('json', data_files=data_files)
        self.dataset['train'] = self.dataset['train'].map(self._preprocess_data, batched=True)
        self.dataset['valid'] = self.dataset['valid'].map(self._preprocess_data, batched=True)
        self.dataset['test'] = self.dataset['test'].map(self._preprocess_data, batched=True)

    def _train_model(self, config, trial=None):
        model_config = config.get('model')
        peft_config = config.get('peft_config')

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_NAME,
            max_seq_length=model_config.get('max_length'),
            load_in_4bit=model_config.get('load_in_4bit'),
            load_in_8bit=model_config.get('load_in_8bit'),
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=peft_config.get('r'), 
            finetune_mlp_modules=peft_config.get('finetune_mlp_modules'),  
            target_modules=peft_config.get('target_modules'), 
            lora_alpha=peft_config.get('lora_alpha'),  
            lora_dropout=peft_config.get('lora_dropout'),  
            bias=peft_config.get('bias'), 
            random_state=peft_config.get('bias'), 
            use_rslora=peft_config.get('use_slora'),  
        )

        args = SFTConfig(
            dataset_text_field=model.get("dataset_text_field"),
            per_device_train_batch_size=model.get("per_device_train_batch_size"),
            gradient_accumulation_steps=model.get("gradient_accumulation_steps"),
            warmup_steps=model.get("warmup_steps"),
            num_train_epochs=model.get("num_train_epochs"),
            max_steps=model.get("max_steps"),
            learning_rate=model.get("learning_rate"),
            logging_steps=model.get("logging_steps"),
            optim=model.get("optim"),
            weight_decay=model.get("weight_decay"),
            lr_scheduler_type=model.get("lr_scheduler_type"),
            seed=model.get("seed"),
            report_to=model.get("report_to")
        )


        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset['train'],
            eval_dataset = dataset['valid'],
            args=args
        )

        train_results = trainer.train()
        eval_results = trainer.evaluate()
        
        del model
        del trial
        gc.collect()
        torch.cuda.empty_cache()

        return {
            'train_results' : train_results,
            'eval_results'  : eval_results
        }

    def _run_trial(self):
        study = optuna.create_study(
            direction='minimize',
            sampler=GPSampler(),
            pruner=optuner.Pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5)
        )
        
        try:
            best_parameter = study.optimize(
                lambda trial: self._train_model(self.default_config, trial=trial),
                n_trials=6,
                catch=(Exception)
            )
        except KeyboardInterrupt:
            print("Keyboard Inturrupt")
            
    def train(self):
        if self.default_config.get('hyperparameter_search'):
            self._run_trial(self.default_config)
        else:
            self._train_model(self.default_config)