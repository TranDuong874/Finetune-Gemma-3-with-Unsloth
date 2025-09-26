import yaml
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')

    args = parser.parse_args()
    print(f'Current config used: {args}')
    with open(args, 'r') as file:
        config_file = yaml.safe_load(file)
    
