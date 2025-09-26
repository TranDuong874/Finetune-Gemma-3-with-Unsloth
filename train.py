import yaml
import argparse
from gemma3_finetuner import Gemma3Finetuner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Model to train')
    parser.add_argument('-c', '--config', help='Config file path')
    
    MODELS = {
        'gemma-3-it': 'configs/gemma-3-it.yaml'
    }
    
    args = parser.parse_args()
    
    # Use specified config or model's default config
    config_path = args.config if args.config else MODELS.get(args.model)
    
    if not config_path:
        print(f'Error: Unknown model "{args.model}". Available models: {list(MODELS.keys())}')
        exit(1)
    
    print(f'Model: {args.model}, Config: {config_path}')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if args.model == 'gemma-3-it':
        finetuner = Gemma3Finetuner(config)
        results = finetuner.train()
        print(f'Training completed. Results: {results}')
    else:
        print(f'Trainer for "{args.model}" not implemented yet.')