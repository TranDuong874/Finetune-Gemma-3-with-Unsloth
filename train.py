import yaml
import argparse
from gemma_3_trainer import Gemma3InstructTrainer
import torch.distributed as dist

def setup_distributed_top():
    """Initialize PG and set CUDA device. MUST run before libraries that call dist."""
    is_dist = False
    rank = 0
    local_rank = 0
    world_size = 1

    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        is_dist = True

        if not dist.is_initialized():
            # helpful defaults
            os.environ.setdefault("NCCL_DEBUG", os.environ.get("NCCL_DEBUG", "WARN"))
            dist.init_process_group(backend="nccl", init_method="env://")

        # map LOCAL_RANK to CUDA ordinal respecting CUDA_VISIBLE_DEVICES
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            arr = [int(x) for x in cvd.split(",") if x.strip() != ""]
            if local_rank >= len(arr):
                raise RuntimeError(f"LOCAL_RANK {local_rank} >= len(CUDA_VISIBLE_DEVICES) {len(arr)}")
            cuda_ordinal = arr[local_rank]
        else:
            cuda_ordinal = local_rank

        torch.cuda.set_device(cuda_ordinal)
    else:
        # single-process fallback
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    print(f"[setup_distributed_top] rank={rank} local_rank={local_rank} world_size={world_size} is_dist={is_dist} device={torch.cuda.current_device()}", flush=True)
    return rank, local_rank, world_size, is_dist

RANK, LOCAL_RANK, WORLD_SIZE, IS_DIST = setup_distributed_top()

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
        finetuner = Gemma3InstructTrainer(config)
        results = finetuner.train()
        print(f'Training completed. Results: {results}')
    else:
        print(f'Trainer for "{args.model}" not implemented yet.')