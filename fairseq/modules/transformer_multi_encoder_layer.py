# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadCrossAttention
)


class TransformerMultiEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,   #need to add the embedding dimentions of other modalities
        qdim :float=768,
        kdim : float = 768,
        vdim : float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        export: bool = False,
        self_attention:bool=False,
        encoder_decoder_attention:bool=False
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.query_dim=qdim
        self.key_dim = kdim
        self.value_dim = vdim

        self.self_attention=self_attention
        self.encoder_decorder_attention=encoder_decoder_attention

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.self_attn = MultiheadCrossAttention( 
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=self.self_attention,
            kdim=self.key_dim,
            vdim=self.value_dim,
            qdim=self.query_dim,
            encoder_decoder_attention=self.encoder_decorder_attention
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)  #Final transform
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def forward(   #If we are modifying this we need three modalities
        self,
        # x: torch.Tensor,  #This is dictionary consist of text,audio,video
        x_q: torch.Tensor,   #The modality to calculate the query
        x_k: torch.Tensor,   #The modality to calculate the key
        x_v: torch.Tensor, #The modality to calculate the value (for future)
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,    
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        # residual = x
        
        residual = x_q
        
        # x, attn = self.self_attn(
        #     query=x,
        #     key=x,
        #     value=x,
        #     key_padding_mask=self_attn_padding_mask,
        #     need_weights=False,
        #     attn_mask=self_attn_mask,
        # )

        x, attn = self.self_attn(
            query=x_q,
            key=x_k,
            value=x_v,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        

        #x=x_cr    
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn
        
