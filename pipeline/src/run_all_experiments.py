import pandas as pd
from pathlib import Path
import argparse
import yaml
from src.utils.config import TrainingConfig, setup_experiment_dir
from src.utils.auth import authenticate_huggingface
from src.training.trainer import BindingTrainer

def update_config_file(config_path, glycan_encoder, protein_encoder, binding_predictor):
    """Update the config file with new encoder and predictor values."""
    # use existing yaml file (likely )
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Update the values of the parts of the model to chnage the encoders 
    config_data['glycan_encoder_type'] = glycan_encoder
    config_data['protein_encoder_type'] = protein_encoder
    config_data['binding_predictor_type'] = binding_predictor
    
    # Write back to the file
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    print(f"Updated config file with: {glycan_encoder}-{protein_encoder}-{binding_predictor}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    config_path = args.config
    
    glycan_encoders = ["chemberta", "rdkit", "gnn"]
    protein_encoders = ["biopy", "pt_gnn"]  # Removed "esmc" as it was broken
    binding_predictors = ["dnn", "mean"]
    
    combinations = []
    for glycan_encoder in glycan_encoders:
        for protein_encoder in protein_encoders:
            for binding_predictor in binding_predictors:
                combination = (glycan_encoder, protein_encoder, binding_predictor)
                combinations.append(combination)
    
    for i, (glycan_encoder, protein_encoder, binding_predictor) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}]")
        
        # Update the config file with the current combination
        update_config_file(config_path, glycan_encoder, protein_encoder, binding_predictor)
        
        # Load the updated config
        config = TrainingConfig.from_yaml(config_path)
        
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
        print(f"Starting training for {glycan_encoder}-{protein_encoder}-{binding_predictor}...")
        trainer = BindingTrainer(config)
        trainer.train(predict_df, glycans_df, proteins_df)
    
    print("All combinations completed!")

if __name__ == "__main__":
    main()