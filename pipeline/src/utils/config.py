from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml
from pathlib import Path
from datetime import datetime

@dataclass
class TrainingConfig:
    # required parameters
    output_dir: str
    predict_data_path: str
    glycans_data_path: str
    proteins_data_path: str
    glycan_encoder_type: str
    protein_encoder_type: str
    binding_predictor_type: str
<<<<<<< HEAD
    glycan_type: str
=======
>>>>>>> main
    num_epochs: int
    batch_size: int
    learning_rate: float
    checkpoint_frequency: int
    random_state: int
    log_predict: bool
    train_final_model: bool
    use_kfold: bool
    split_mode: str
    # optional parameters
    val_split: float = 0.2
    k_folds: int = 1
    device: str = "cuda"  # or "cpu"
    model_specific_params: Dict = field(default_factory=dict)
    hf_auth: Optional[bool] = None

    # Based on: https://github.com/run-llama/llama_deploy/blob/f625bfd506f1be36349b23e692ca7b976f39f636/examples/llamacloud/google_drive/src/workflow.py
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, save_path: str):
        with open(save_path, 'w') as f:
            yaml.dump(self.__dict__, f)


# Generated with GitHub Copilot
def setup_experiment_dir(config: TrainingConfig) -> Path:
    """
    Create experiment directory with timestamp and copy config
    
    Returns:
        Path: Path to experiment directory
    """
    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{timestamp}_{config.glycan_encoder_type}_{config.protein_encoder_type}"
    
    # Create experiment directory
    exp_dir = Path(config.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    
    # Save config
    config.save(exp_dir / "config.yaml")
    
    return exp_dir