import torch
import torch.nn as nn

class ClassicalTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src):
        seq_len = src.size(1)
        # Ensure sequence length doesn't exceed position encoding
        if seq_len > self.pos_encoder.size(1):
             # Truncate or handle error - for now, simplistic handling
             pass 
        
        src = self.embedding(src) + self.pos_encoder[:, :seq_len, :]
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
