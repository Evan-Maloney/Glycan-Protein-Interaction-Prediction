import pandas as pd
from pathlib import Path
import argparse
from src.utils.config import TrainingConfig, setup_experiment_dir
from src.utils.auth import authenticate_huggingface
from src.training.trainer import BindingTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)

    if config.hf_auth:
        authenticate_huggingface()
    
    data_path = Path(config.data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    exp_dir = setup_experiment_dir(config)
    config.output_dir = str(exp_dir)
    data_df = pd.read_csv(config.data_path)
    print(f"Loaded {len(data_df)} samples from {config.data_path}")
    
    # run the training experiment
    print("Starting training...")
    trainer = BindingTrainer(config)
    trainer.train(data_df, config.precomputed_path)

if __name__ == "__main__":
    main()