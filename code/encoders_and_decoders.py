import torch.nn as nn
import torch
from abc import ABC, abstractmethod
from embeddings import TraceToEmbedding
from output_layers import IndependentOutputLayer, DependentOutputLayer
from utils import AddNorm, FeedForwardBlock


class TransformerEncoder(nn.Module):
    def __init__(self, 
        embedding_dim : int,
        attn_heads : int = 2, 
        num_layers : int = 1, 
        dropout : float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attn_heads = attn_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=self.embedding_dim,
                attn_heads=self.attn_heads,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
            ])

        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, 
                 embedding_dim : int, 
                 attn_heads : int = 2,
                 dropout : float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.attn_heads = attn_heads
        self.hidden_dim = self.embedding_dim * 4
        self.dropout = dropout

        assert self.embedding_dim % self.attn_heads == 0, "Embedding dimension must be divisible by number of attention heads"
        self.attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.attn_heads, dropout=self.dropout, batch_first=True)
        self.add_norm1 = AddNorm(self.embedding_dim, dropout=self.dropout)

        self.feed_forward = FeedForwardBlock(self.embedding_dim, self.hidden_dim)
        self.add_norm2 = AddNorm(self.embedding_dim, dropout=self.dropout)



    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        Q, K, V = x, x, x
        attn_output, _ = self.attention(Q, K, V, key_padding_mask=mask)
        x = self.add_norm1(x, attn_output)

        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)

        return x



class TransformerDecoder(nn.Module):
    def __init__(self, 
        embedding_dim : int,
        attn_heads : int = 2, 
        num_layers : int = 1, 
        dropout : float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attn_heads = attn_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dim=self.embedding_dim,
                attn_heads=self.attn_heads,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
            ])

        
    def forward(self, y, encoder_output, tgt_padding_mask=None, src_padding_mask=None):
        for layer in self.layers:
            y = layer(y, encoder_output, tgt_padding_mask=tgt_padding_mask, src_padding_mask=src_padding_mask)

        return y
    


class TransformerDecoderBlock(nn.Module):
    def __init__(self, 
                 embedding_dim : int, 
                 attn_heads : int = 2,
                 dropout : float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.attn_heads = attn_heads
        self.hidden_dim = self.embedding_dim * 4
        self.dropout = dropout

        assert self.embedding_dim % self.attn_heads == 0, "Embedding dimension must be divisible by number of attention heads"

        self.masked_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.attn_heads, dropout=self.dropout, batch_first=True)
        self.add_norm1 = AddNorm(emb_size=self.embedding_dim, dropout=self.dropout)

        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.attn_heads, dropout=self.dropout, batch_first=True)
        self.add_norm2 = AddNorm(emb_size=self.embedding_dim, dropout=self.dropout)

        self.feed_forward = FeedForwardBlock(self.embedding_dim, self.hidden_dim)
        self.add_norm3 = AddNorm(emb_size=self.embedding_dim, dropout=self.dropout)


    def _generate_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask


    def forward(self, y, encoder_output, tgt_padding_mask=None, src_padding_mask=None):
        Q, K, V = y, y, y
        subsequent_mask = self._generate_subsequent_mask(y.size(1)).to(y.device)
        masked_attn_output, _ = self.masked_attention(Q, K, V, attn_mask=subsequent_mask, key_padding_mask=tgt_padding_mask)
        y_1 = self.add_norm1(y, masked_attn_output)

        Q, K, V = y_1, encoder_output, encoder_output
        attn_output, _ = self.cross_attention(Q, K, V, key_padding_mask=src_padding_mask)
        y_2 = self.add_norm2(y_1, attn_output)

        ff_output = self.feed_forward(y_2)
        y_2 = self.add_norm3(y_2, ff_output)

        return y_2


################################################################################################

class StepEventTransformer(nn.Module, ABC):
    def __init__(self,
                 cat_attributes : list, # list of dictionary with attribute info
                 num_attributes : list, # list of numerical attribute name
                 embedding_size : int = 50,
                 emb_output_size : int | None = None,
                 feature_attn_heads : int = 0,
                 encoder_layers : int = 4,
                 decoder_layers : int = 0,
                 encoder_attn_heads : int = 2,
                 dropout : float = 0.1,
                 independent_output : bool = False,
                 ):
        super(StepEventTransformer, self).__init__()

        if decoder_layers > 0:
            print(f"[WARNING] StepEventTransformer does not use decoder")

        self.cat_attributes = cat_attributes
        self.num_attributes = num_attributes

        self.emb_size = embedding_size
        self.encoder_blocks = encoder_layers
        self.decoder_blocks = 0

        self._first_pad = cat_attributes[0]['pad']

        self.embeddings = TraceToEmbedding(
                            self.cat_attributes,
                            self.num_attributes,
                            embedding_size = embedding_size,
                            emb_output_size = emb_output_size,
                            feature_attn_heads = feature_attn_heads
                            )
        
        self.hidden_size = self.embeddings.output_size
        
        self.encoder = TransformerEncoder(embedding_dim=self.hidden_size, attn_heads=encoder_attn_heads, num_layers=encoder_layers, dropout=dropout)
        
        self.independent_output = independent_output

        output_layer_class = IndependentOutputLayer if self.independent_output else DependentOutputLayer
        self.output_layer = output_layer_class(
            vocab_sizes=[attr['dict_size'] for attr in self.cat_attributes],
            latent_size=self.hidden_size,
        )


    @property
    def device(self):
        return next(self.parameters()).device


    def forward(self, x_cat : torch.Tensor, x_num : torch.Tensor = None, return_output=False):
        embeddings = self.embeddings(x_cat, x_num)
        encoder_mask = (x_cat[:, 0, :] == self._first_pad)
        encoder_output = self.encoder(embeddings, mask=encoder_mask)

        #encoder_logits = encoder_output.mean(dim=1)
        encoder_logits = encoder_output[:, -1, : ]

        logits = self.output_layer(encoder_logits)

        if return_output:
            return logits, encoder_output

        return logits

    def encoder_output(self, x_cat : torch.Tensor, x_num : torch.Tensor = None, return_output=False):
        embeddings = self.embeddings(x_cat, x_num)
        encoder_mask = (x_cat[:, 0, :] == self._first_pad)
        encoder_output = self.encoder(embeddings, mask=encoder_mask)

        return encoder_output[:, -1, : ]

    
    def predict(self, x_cat : torch.Tensor, x_num : torch.Tensor = None):
        logits = self.forward(x_cat, x_num)
        predictions = torch.stack([logit.argmax(dim=1) for logit in logits], dim=1)

        return predictions



class EventTransformer(nn.Module, ABC):
    def __init__(self,
                 cat_attributes : list, # list of dictionary with attribute info
                 num_attributes : list, # list of numerical attribute name
                 embedding_size : int = 50,
                 emb_output_size : int | None = None,
                 feature_attn_heads : int = 0,
                 encoder_layers : int = 4,
                 decoder_layers : int = 4,
                 encoder_attn_heads : int = 2,
                 dropout : float = 0.1,
                 independent_output : bool = False,
                 shared_embeddings : bool = False,
                 complex_output_layer : bool = False,
                 ):
        super(EventTransformer, self).__init__()

        self.cat_attributes = cat_attributes
        self.num_attributes = num_attributes

        self.emb_size = embedding_size
        self.encoder_blocks = encoder_layers
        self.decoder_blocks = decoder_layers
        self.complex_output_layer = complex_output_layer

        self._first_pad = cat_attributes[0]['pad']

        self.embeddings = TraceToEmbedding(
                            self.cat_attributes,
                            self.num_attributes,
                            embedding_size = embedding_size,
                            emb_output_size = emb_output_size,
                            feature_attn_heads = feature_attn_heads
                            )

        if shared_embeddings:
            self.decoder_embeddings = self.embeddings
        else:
            self.decoder_embeddings = TraceToEmbedding(
                                self.cat_attributes,
                                self.num_attributes,
                                embedding_size = embedding_size,
                                emb_output_size = emb_output_size,
                                feature_attn_heads = feature_attn_heads
                                )
        
        
        self.hidden_size = self.embeddings.output_size
        
        self.encoder = TransformerEncoder(embedding_dim=self.hidden_size, attn_heads=encoder_attn_heads, num_layers=encoder_layers, dropout=dropout)
        self.decoder = TransformerDecoder(embedding_dim=self.hidden_size, attn_heads=encoder_attn_heads, num_layers=decoder_layers, dropout=dropout)

        self.independent_output = independent_output

        output_layer_class = IndependentOutputLayer if self.independent_output else DependentOutputLayer
        self.output_layer = output_layer_class(
            vocab_sizes=[attr['dict_size'] for attr in self.cat_attributes],
            latent_size=self.hidden_size,
            complex=self.complex_output_layer
        )

        


    @property
    def device(self):
        return next(self.parameters()).device


    def forward(self, x_cat : torch.Tensor, x_num : torch.Tensor = None, y_cat : torch.Tensor = None, y_num : torch.Tensor = None):
        encoder_output, encoder_mask = self.encode(x_cat, x_num)
        decoder_output = self.decode(encoder_output, encoder_mask, y_cat, y_num)

        logits = self.output_layer(decoder_output)

        return logits
    
    def logits_and_decoder_output(self, x_cat : torch.Tensor, x_num : torch.Tensor = None, y_cat : torch.Tensor = None, y_num : torch.Tensor = None):
        encoder_output, encoder_mask = self.encode(x_cat, x_num)
        decoder_output = self.decode(encoder_output, encoder_mask, y_cat, y_num)

        logits = self.output_layer(decoder_output)

        return logits, decoder_output


    def decoder_output(self, x_cat : torch.Tensor, x_num : torch.Tensor = None, y_cat : torch.Tensor = None, y_num : torch.Tensor = None):
        encoder_output, encoder_mask = self.encode(x_cat, x_num)
        decoder_output = self.decode(encoder_output, encoder_mask, y_cat, y_num)

        return decoder_output
    

    def encode(self, x_cat : torch.Tensor, x_num : torch.Tensor = None):
        embeddings = self.embeddings(x_cat, x_num)
        encoder_mask = (x_cat[:, 0, :] == self._first_pad)
        encoder_output = self.encoder(embeddings, mask=encoder_mask)

        return encoder_output, encoder_mask

    def decode(self, encoder_output : torch.Tensor, encoder_mask : torch.Tensor, y_cat : torch.Tensor = None, y_num : torch.Tensor = None):
        embeddings_d = self.decoder_embeddings(y_cat, y_num)
        decoder_mask = (y_cat[:, 0, :] == self._first_pad)
        decoder_output = self.decoder(embeddings_d, encoder_output, tgt_padding_mask=decoder_mask, src_padding_mask=encoder_mask)
        return decoder_output


    def predict(self, encoder_output : torch.Tensor, encoder_mask : torch.Tensor, y_cat : torch.Tensor = None, y_num : torch.Tensor = None):
        decoder_output = self.decode(encoder_output, encoder_mask, y_cat, y_num)
        logits = self.output_layer(decoder_output)

        predictions = torch.stack([logit[:, -1, :].argmax(dim=1) for logit in logits], dim=1)
        return predictions
    
