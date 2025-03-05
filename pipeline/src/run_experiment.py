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
    
    
    exp_dir = setup_experiment_dir(config)
    config.output_dir = str(exp_dir)
    
    
    predict_data_path = Path(config.predict_data_path)
    predict_data_path.parent.mkdir(parents=True, exist_ok=True)
    predict_df = pd.read_csv(config.predict_data_path, sep="\t")
    print(f"Loaded predict df: {len(predict_df)} samples from {config.predict_data_path}")
    
    glycans_data_path = Path(config.glycans_data_path)
    glycans_data_path.parent.mkdir(parents=True, exist_ok=True)
    glycans_df = pd.read_csv(config.glycans_data_path, sep="\t")
    print(f"Loaded glycans df: {len(glycans_df)} samples from {config.glycans_data_path}")
    
    proteins_data_path = Path(config.proteins_data_path)
    proteins_data_path.parent.mkdir(parents=True, exist_ok=True)
    proteins_df = pd.read_csv(config.proteins_data_path, sep="\t")
    print(f"Loaded proteins df: {len(proteins_df)} samples from {config.proteins_data_path}")
    
    # run the training experiment
    print("Starting training...")
    trainer = BindingTrainer(config)
    trainer.train(predict_df, glycans_df, proteins_df)

if __name__ == "__main__":
    main()