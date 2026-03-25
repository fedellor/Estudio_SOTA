import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):#, device: str = 'cpu'):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        #self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)



class AddNorm(nn.Module):
    def __init__(self, emb_size, dropout=0.2):#, device='cpu'):
        super().__init__()

        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

        #? Se puede probar Pre-LN: https://arxiv.org/abs/2002.04745
        #? Norm -> Sublayer -> Dropout -> Add

        #self.to(device)

    def forward(self, x1, x2):
        add = x1 + self.dropout(x2)
        norm = self.layer_norm(add)

        return norm



class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, hidden_size, inicialize_weights=False):#, device = 'cpu'):
        super().__init__()

        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, emb_size)
        self.relu = nn.ReLU()

        if inicialize_weights:
            init_linear(self.fc1)
            init_linear(self.fc2)

        #self.to(device)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        out = self.fc2(h)

        return out
    

class GatedFeedForwardBlock(nn.Module):
    def __init__(self, emb_size, hidden_size, inicialize_weights=False):#, device = 'cpu'):
        super().__init__()

        self.fc_gate = nn.Linear(emb_size, hidden_size)
        self.fc_value = nn.Linear(emb_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, emb_size)
        self.act = nn.GELU() 

        if inicialize_weights:
            init_linear(self.fc_gate)
            init_linear(self.fc_value)
            init_linear(self.fc_output)

    def forward(self, x):

        gate = self.act(self.fc_gate(x))
        value = self.fc_value(x)
        h = gate * value
        out = self.fc_output(h)

        return out



import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class MultiheadAttentionWithRoPE(nn.Module):
    """
    Multi-Head Attention con Rotary Position Embeddings.
    USO EXCLUSIVO PARA SELF-ATTENTION (Encoder / Decoder).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, 
                 kdim=None, vdim=None,
                 batch_first=False, max_seq_len=2048, rope_base=10000, inicialize_weights=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert embed_dim // num_heads % 2 == 0, "head_dim must be even for Rotary Positional Embeddings"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        
        # Dimensiones de K y V
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        # Proyecciones lineales
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # RoPE (Incondicional, porque es Self-Attention)
        #todo AJUSTAR EL TEMA DE LAS DIMENSIONES. NO VA BIEN.
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len, rope_base)
        self.dropout = nn.Dropout(dropout)
        
        if inicialize_weights:
            self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)
        
        # 1. Proyecciones
        Q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. RoPE (Siempre activado para Self-Attention)
        cos_q, sin_q = self.rotary_emb(tgt_len, query.device)
        cos_k, sin_k = self.rotary_emb(src_len, key.device)
        
        # Ajuste de dimensiones para broadcasting (1, 1, seq, dim)
        cos_q, sin_q = cos_q[None, None, :, :], sin_q[None, None, :, :]
        cos_k, sin_k = cos_k[None, None, :, :], sin_k[None, None, :, :]
        
        Q = apply_rotary_pos_emb(Q, cos_q, sin_q)
        K = apply_rotary_pos_emb(K, cos_k, sin_k)
        
        
        final_mask = None
        
        # A) Attn Mask (Causal o Custom)
        if attn_mask is not None:
            # Asegurar dimensiones (1, 1, Tgt, Src)
            if attn_mask.dim() == 2:
                current_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                current_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)
            else:
                current_mask = attn_mask

            # Creamos máscara de float (0.0) y rellenamos con -inf donde tu máscara es True
            final_mask = torch.zeros_like(current_mask, dtype=query.dtype)
            final_mask = final_mask.masked_fill(current_mask, float('-inf'))
        
        # B) Padding Mask
        if key_padding_mask is not None:
            # (Batch, Src) -> (Batch, 1, 1, Src)
            padding_mask_additive = torch.zeros((batch_size, 1, 1, src_len), 
                                                device=query.device, 
                                                dtype=query.dtype)
            # Rellenamos con -inf donde key_padding_mask es True
            padding_mask_additive = padding_mask_additive.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), 
                float('-inf')
            )
            
            # Sumar máscaras (acumular los -inf)
            if final_mask is None:
                final_mask = padding_mask_additive
            else:
                final_mask = final_mask + padding_mask_additive
        
        # 4. Atención
        # Ponemos is_causal=False porque ya hemos construido la máscara causal manualmente arriba
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=final_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False 
        )
        
        # Salida
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        return attn_output, None



def init_linear(layer):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)



def get_windows_from_traces(traces : torch.Tensor, current_lengths : torch.Tensor, window_size : int):
    device = traces.device
    B, A, _ = traces.shape

    end = torch.clamp(current_lengths, min=window_size)

    start = end - window_size

    # shape: (B, window_size)
    offsets = torch.arange(window_size, device=device)
    indices = start.unsqueeze(1) + offsets.unsqueeze(0)

    # expand for attributes
    indices = indices.unsqueeze(1).expand(-1, A, -1)

    return torch.gather(traces, dim=2, index=indices)



def update_traces_with_predictions(traces, predictions, current_lengths):
    B, A, T = traces.shape

    if torch.any(current_lengths >= T):
        raise RuntimeError(
            f"Write position out of bounds: max={current_lengths.max().item()}, T={T}"
        )

    idx = current_lengths.unsqueeze(1).expand(-1, A)
    traces[torch.arange(B).unsqueeze(1), torch.arange(A), idx] = predictions
    return traces