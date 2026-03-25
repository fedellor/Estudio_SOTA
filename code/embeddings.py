import torch.nn as nn
import torch
from utils import PositionalEncoding

class TraceToEmbedding(nn.Module):
    def __init__(self, 
                 cat_attributes : list, 
                 num_attributes : list = [], 
                 embedding_size : int = 32, 
                 emb_output_size : int | None = None, 
                 feature_attn_heads : int = 0):
        super().__init__()

        self.attributes = cat_attributes
        self.attributes_num = num_attributes
        self.embedding_size = embedding_size

        self.feature_attn_heads = feature_attn_heads

        if emb_output_size is None:
            self.project = False
            self.output_size = self.embedding_size * (len(self.attributes) + len(self.attributes_num))
        else:
            self.project = True
            self.output_size = emb_output_size

        self.embeddings = nn.ModuleList([
            nn.Embedding(attr['dict_size'], embedding_size, padding_idx=attr['pad']) for attr in self.attributes
            ])
        
        if len(self.attributes_num) > 0:
            self.numeric_projections = nn.ModuleList([
                nn.Linear(1, embedding_size) for _ in range(len(self.attributes_num))
            ])

        self.embedding_dim = embedding_size * (len(self.attributes) + len(self.attributes_num))
        
        if self.feature_attn_heads >= 1:
            assert self.embedding_dim % self.feature_attn_heads == 0, "Embedding dimension must be divisible by number of attention heads"
            self.feature_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.feature_attn_heads, batch_first=True)

        if self.project:
            self.projection = nn.Linear(self.embedding_dim, self.output_size)

        self.pos_enc = PositionalEncoding(d_model=self.output_size)


    @property
    def device(self):
        return next(self.parameters()).device

    
    def forward(self, categorical_attrs, numeric_attrs=None):
        embedded_attrs = []

        for attr_index in range(len(self.attributes)):
            embedded_attr = self.embeddings[attr_index](categorical_attrs[:, attr_index, :].to(self.device))
            embedded_attrs.append(embedded_attr)
        
        if len(self.attributes_num) > 0 and numeric_attrs is not None:
            for i in range(len(self.attributes_num)):
                numeric_attr = numeric_attrs[:, i, :].unsqueeze(-1).to(self.device).type(torch.float32)
                projected_numeric_attr = self.numeric_projections[i](numeric_attr)
                embedded_attrs.append(projected_numeric_attr)
        
        combined_embeddings = torch.cat(embedded_attrs, dim=-1)

        if self.feature_attn_heads >= 1:
            attn_output, _ = self.feature_attention(combined_embeddings, combined_embeddings, combined_embeddings)
        else:
            attn_output = combined_embeddings

        projected_output = self.projection(attn_output) if self.project else attn_output

        projected_output = self.pos_enc(projected_output)

        return projected_output
        