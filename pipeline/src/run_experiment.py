import pandas as pd
from pathlib import Path
import argparse
from src.utils.config import TrainingConfig, setup_experiment_dir
from src.utils.auth import authenticate_huggingface
from src.training.trainer import BindingTrainer, ExperimentTracker  # Import ExperimentTracker from trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--show-all-results", action="store_true", help="Display results from all previous experiments")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)

    if config.hf_auth:
        authenticate_huggingface()
    
    exp_dir = setup_experiment_dir(config)
    config.output_dir = str(exp_dir)
    
    print("\n=== Current Experiment Configuration ===")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("="*40)
    
    if args.show_all_results and not config.output_dir:
        experiment_base_dir = Path("experiments")
        tracker = ExperimentTracker(experiment_base_dir)
        results = tracker.get_results_table()
        if not results.empty:
            print("\n=== All Experiment Results ===")
            print(results.to_string())
        else:
            print("No experiment results found.")
        return
    
    # Load data for training
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
    
    # Run the training experiment
    print("Starting training...")
    trainer = BindingTrainer(config)
    trainer.train(predict_df, glycans_df, proteins_df)
    
    # After training, show all experiment results if requested
    if args.show_all_results:
        print("\n=== Complete Experiment Results Table ===")
        base_dir = Path(config.output_dir).parent
        tracker = ExperimentTracker(base_dir)
        all_results = tracker.get_results_table()
        
        # Format the results table for better readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        # Sort by validation loss (ascending)
        sorted_results = all_results.sort_values('best_val_loss')
        
        # Display results
        print(sorted_results.to_string())
        
        # Reset pandas display options
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()