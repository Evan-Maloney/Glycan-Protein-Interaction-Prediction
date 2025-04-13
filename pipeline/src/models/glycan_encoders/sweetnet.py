# REFERENCES
# https://github.com/BojarLab/glycowork/blob/master/glycowork/ml/inference.py

import torch
from typing import List
from ...base.encoders import GlycanEncoder
from glycowork.ml.models import prep_model
from glycowork.ml.processing import dataset_to_dataloader

class SweetNetEncoder(GlycanEncoder):
    def __init__(self):
        super().__init__()
        self.model = prep_model("SweetNet", 1011, trained = True)
        self.model.eval()
        self._embedding_dim = self.model.item_embedding.embedding_dim
        
    def _process_iupacs(self, glycans: List[str]) -> List[str]:
        for count, iupac in enumerate(glycans): 
             # remove the edning link that may be at the end of an iupac string
            last_open_bracket_idx = iupac.rfind('(')
            last_closed_bracket_idx = iupac.rfind(')')
            if last_open_bracket_idx > last_closed_bracket_idx:
                iupac = iupac[:last_open_bracket_idx]
                
            # format sulfate/phosphate groups
            while 'O' in iupac:
                oxygen_idx = iupac.find('O')
                ox_modification_idx = oxygen_idx + 1
                ox_modification = iupac[ox_modification_idx]
                
                end_idx = ox_modification_idx + 1
                while(end_idx < len(iupac) and iupac[end_idx].isnumeric()):
                    end_idx += 1
                num_groups = int(iupac[ox_modification_idx + 1: end_idx]) if ox_modification_idx + 1 < end_idx else 1

                positions = []
                start_idx = oxygen_idx
                for i in range(num_groups):
                    start_idx -= 1
                    position_end = start_idx
                    while iupac[start_idx].isnumeric():
                        start_idx -= 1
                    positions.insert(0, iupac[start_idx + 1: position_end + 1])
                start_idx += 1
                replacement = "".join(f"{position}{ox_modification}" for position in positions)
                iupac = iupac.replace(iupac[start_idx : end_idx], replacement)
            
            iupac = iupac.replace('MDPLys', '')
                
            glycans[count] = iupac
        return glycans
    
    def encode_iupac(self, iupac: str, device: torch.device) -> torch.Tensor:
        iupac = self._process_iupacs([iupac])[0]
        glycan_loader = dataset_to_dataloader([iupac], range(1), shuffle = False)
        res = torch.zeros((1, 128), device=device)
        # Get predictions for each mini-batch
        idx = 0
        for data in glycan_loader:
            x, y, edge_index, batch = data.labels, data.y, data.edge_index, data.batch
            x, y, edge_index, batch = x.to(device), y.to(device), edge_index.to(device), batch.to(device)
            with torch.no_grad():
                # outputs prediction for the classification task it was trained on, and the intermediate glycan embedding
                pred, out = self.model(x, edge_index, batch, inference = True)
                
            batch_size = out.size(0)
            res[idx:idx + batch_size] = out
            idx += batch_size
        return res
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        batch_data = self._process_iupacs(batch_data)
        glycan_loader = dataset_to_dataloader(batch_data, range(len(batch_data)), shuffle = False)
        res = torch.zeros((len(batch_data), 128), device=device)
        # Get predictions for each mini-batch
        idx = 0
        for data in glycan_loader:
            x, y, edge_index, batch = data.labels, data.y, data.edge_index, data.batch
            x, y, edge_index, batch = x.to(device), y.to(device), edge_index.to(device), batch.to(device)
            with torch.no_grad():
                # outputs prediction for the classification task it was trained on, and the intermediate glycan embedding
                pred, out = self.model(x, edge_index, batch, inference = True)
                
            batch_size = out.size(0)
            res[idx:idx + batch_size] = out
            idx += batch_size
        return res
    
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        """Not implemented since this model does not use SMILES encoding."""
        raise NotImplementedError("SMILES encoding is not supported in SweetNetGlycanEncoder.")
    
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def to(self, device):
        self.model = self.model.to(device)
        return self