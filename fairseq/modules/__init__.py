# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .adaptive_input import AdaptiveInput
from .adaptive_softmax import AdaptiveSoftmax
from .beamable_mm import BeamableMM
from .character_token_embedder import CharacterTokenEmbedder
from .conv_tbc import ConvTBC
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from .dynamic_convolution import DynamicConv, DynamicConv1dTBC
#from .dynamicconv_layer import DynamicconvLayer
from .gelu import gelu, gelu_accurate
from .grad_multiply import GradMultiply
from .highway import Highway
from .layer_norm import LayerNorm
from .learned_positional_embedding import LearnedPositionalEmbedding
from .learned_positional_embedding_mul import LearnedPositionalEmbeddingMul
from .lightweight_convolution import LightweightConv, LightweightConv1dTBC
#from .lightconv_layer import LightconvLayer
from .linearized_convolution import LinearizedConvolution
from .logsumexp_moe import LogSumExpMoE
from .mean_pool_gating_network import MeanPoolGatingNetwork
from .multihead_attention import MultiheadAttention
from .multihead_cross_attention import MultiheadCrossAttention
from .positional_embedding import PositionalEmbedding
from .positional_embedding_mul import PositionalEmbeddingMul
from .scalar_bias import ScalarBias
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .sinusoidal_positional_embedding_mul import SinusoidalPositionalEmbeddingMul
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .transformer_multi_encoder_layer import TransformerMultiEncoderLayer

from .transformer_wav2vec_encoder_layer import TransformerWav2VecEncoderLayer

from .transformer_sentence_encoder import TransformerSentenceEncoder
from .transformer_multi_encoder import TransformerMultiEncoder

from .transformer_wav2vec_encoder import TransformerWav2VecEncoder

from .unfold import unfold1d
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from .vggblock import VGGBlock

__all__ = [
    'AdaptiveInput',
    'AdaptiveSoftmax',
    'BeamableMM',
    'CharacterTokenEmbedder',
    'ConvTBC',
    'DownsampledMultiHeadAttention',
#    'DyamicconvLayer',
    'DynamicConv1dTBC',
    'DynamicConv',
    'gelu',
    'gelu_accurate',
    'GradMultiply',
    'Highway',
    'LayerNorm',
    'LearnedPositionalEmbedding',
#    'LightconvLayer',
    'LightweightConv1dTBC',
    'LightweightConv',
    'LinearizedConvolution',
    'LogSumExpMoE',
    'MeanPoolGatingNetwork',
    'MultiheadAttention',
    'MultiheadCrossAttention',
    'PositionalEmbedding',
    'ScalarBias',
    'SinusoidalPositionalEmbedding',
    'TransformerSentenceEncoderLayer',
    'TransformerWav2VecEncoderLayer',
    'TransformerSentenceEncoder',
    'TransformerWav2VecEncoder',
    'TransformerDecoderLayer',
    'TransformerEncoderLayer',
    'VGGBlock',
    'unfold1d',
]
