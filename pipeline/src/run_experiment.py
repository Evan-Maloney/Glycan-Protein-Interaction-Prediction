import pandas as pd
from pathlib import Path
import argparse
from src.utils.config import TrainingConfig, setup_experiment_dir
from src.utils.auth import authenticate_huggingface
from src.training.trainer import BindingTrainer
from torch import nn

# had to add this class here so that sweet talk glycan encoder could run
#bidirectional, two-layered LSTM without padding
class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes, n_layers = 2):
    super(RNN,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.n_layers = n_layers
    
    # Move BatchNorm1d here, use hidden_size (features), not batch_size
    #self.bn1 = nn.BatchNorm1d(hidden_size)
    self.bn1 = nn.BatchNorm1d(29)
    
    self.encoder = nn.Embedding(input_size, hidden_size, padding_idx = self.num_classes-1)
    self.decoder = nn.Linear(hidden_size, num_classes)
    self.gru = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional = True)
    self.logits_fc = nn.Linear(2*hidden_size, num_classes)   
    
  def forward(self, input_seq, input_seq_len, hidden=None):
    #print(f"input_seq shape: {input_seq.shape}")  # Debugging shape
    
    
    embedded = self.encoder(input_seq)  # (seq_len, batch_size, hidden_size)
    #print(f"embedded shape before BN: {embedded.shape}")

    # Permute to (batch_size, hidden_size, seq_len) for BatchNorm1d
    embedded = embedded.permute(1, 2, 0)
    #print(f"embedded shape after permute: {embedded.shape}")

    embedded = self.bn1(embedded)  # This is where the error occurs
    #print(f"embedded shape after BN: {embedded.shape}")

    # Restore to (seq_len, batch_size, hidden_size)
    embedded = embedded.permute(2, 0, 1)

    outputs, (h_n, c_n) = self.gru(embedded, hidden)
    
    logits = self.logits_fc(outputs)
    logits = logits.transpose(0, 1).contiguous()
    logits_flatten = logits.view(-1, self.num_classes)

    return logits_flatten, hidden

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