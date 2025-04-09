import torch
from typing import List
import esm
from ...base.encoders import ProteinEncoder

# used ChatGPT tp modify encode_batch to encode proteins in batches for memory efficiency
class ESM2Encoder(ProteinEncoder):
    def __init__(self):
        super().__init__()
        # self.model = esm.from_pretrained("esm2_t6_8M_UR50D")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self._embedding_dim = self.model.embed_dim
    
    def encode_sequence(self, sequence: tuple) -> torch.Tensor:

        # sample data ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
        # ESM2 has compatibility issues with ESMProtein
        protein = [sequence]

        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        # device = torch.device("cuda")
        # batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[6], return_contacts=True)
        token_representations = results["representations"][6]
        
        embedding = []
        for i, tokens_len in enumerate(batch_lens):
            embedding.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        return embedding[0]
    
    # def encode_batch(self, batch_data: List[str], device) -> torch.Tensor:
    #     # Create all protein objects at once

    #     protein =[]
    #     for i in range(len(batch_data)) :
    #         protein.append(("protein" + str(i), batch_data[i]))

    #     batch_labels, batch_strs, batch_tokens = self.batch_converter(protein)
    #     batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

    #     print(len(batch_lens))
    #     # print(batch_tokens.is_cuda )
    #     batch_tokens = batch_tokens.to(device)

    #     with torch.no_grad():
    #         results = self.model(batch_tokens, repr_layers=[6], return_contacts=True)
    #     token_representations = results["representations"][6]
        
    #     # with torch.cuda.amp.autocast():
    #     #     results = self.model(batch_tokens, repr_layers=[6], return_contacts=True)
    #     # token_representations = results["representations"][6]

    #     embeddings = []
    #     for i, tokens_len in enumerate(batch_lens):
            
    #         embeddings.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    #     # print(embedding[i].shape)
    #     embeddings = torch.vstack(embeddings).cuda()
    #     return embeddings
        
    def encode_batch(self, batch_data: List[str], device, batch_size: int = 4) -> torch.Tensor:
        """Encodes a batch of protein sequences in chunks to prevent out-of-memory errors."""

        # Create all protein objects
        protein = [("protein" + str(i), batch_data[i]) for i in range(len(batch_data))]

        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(device)

        all_embeddings = []  # List to store embeddings

        # Process in chunks
        for i in range(0, len(batch_tokens), batch_size):
            batch_chunk = batch_tokens[i : i + batch_size]  # Get batch slice
            batch_chunk_lens = batch_lens[i : i + batch_size]  # Get corresponding lengths

            with torch.no_grad():
                results = self.model(batch_chunk, repr_layers=[6], return_contacts=True)
            token_representations = results["representations"][6]

            # Extract embeddings for each sequence in the chunk
            for j, tokens_len in enumerate(batch_chunk_lens):
                embedding = token_representations[j, 1 : tokens_len - 1].mean(0)  # Mean pooling
                all_embeddings.append(embedding)

        # Convert list to tensor
        all_embeddings = torch.stack(all_embeddings).to(device)
        return all_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def to(self, device):
        self.model = self.model.to(device)
        return self