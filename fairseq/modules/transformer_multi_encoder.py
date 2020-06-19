# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    PositionalEmbeddingMul,
    TransformerSentenceEncoderLayer,
    TransformerMultiEncoderLayer,
)

import math


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)


class TransformerMultiEncoder(nn.Module):  #We might not need this part since we are already getting embeddings...
    
    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        num_encoder_layers_cross: int = 6,
        embedding_dim: int = 768,
        embedding_dim_text: int = 768,
        embedding_dim_audio: int = 768,
        embedding_dim_video: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seq_len_text: int = 256,  
        max_seq_len_audio: int = 256,
        max_seq_len_video: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        is_start_AV_embeddings: bool = True, 
        offset_positions_by_padding: bool = True, 
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        is_self_attention: bool =True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        is_only_text: bool=False,
        is_only_audio: bool=False,
        is_only_video: bool=False,
        is_all_in: bool=False,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len_t = max_seq_len_text #text
        self.max_seq_len_a = max_seq_len_audio #audio
        self.max_seq_len_v = max_seq_len_video #video
        self.embedding_dim = embedding_dim
        self.embedding_dim_t = embedding_dim_text
        self.embedding_dim_a = embedding_dim_audio
        self.embedding_dim_v = embedding_dim_video
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.is_start_AV_embeddings=is_start_AV_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding

        self.only_t=is_only_text
        self.only_a=is_only_audio
        self.only_v=is_only_video
        self.all_in=is_all_in

        self.embed_scale = embed_scale

       
  
    
        if self.only_v or self.all_in: 

      
            self.SE_embeddings_v = (  #for start and end video #only start so 1
                nn.Embedding(1, self.embedding_dim_v, padding_idx=None)
                if self.is_start_AV_embeddings
                else None
            )

            self.padding_idx_v=1#1
            #Vid2vec max of 5 and dimentions of  256
            self.embed_positions_v = (  #We need 2 postional embeddings matrix for each modality (A,V)
                PositionalEmbeddingMul(
                    self.max_seq_len_v,
                    self.embedding_dim_v,
                    padding_idx=(self.padding_idx_v if offset_positions_by_padding else None),
                    learned=self.learned_pos_embedding,
                )
                if self.use_position_embeddings
                else None
            )

            self.layers_v = nn.ModuleList(  #Text to Audio (The query vector comes from the Text and Key-Value from the Audio)
                [
                    TransformerSentenceEncoderLayer(
                        embedding_dim=self.embedding_dim_v,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        add_bias_kv=add_bias_kv,
                        add_zero_attn=add_zero_attn,
                        export=export,

                    )
                    for _ in range(num_encoder_layers)
                ]
                if is_self_attention
                else None
            )








        if self.only_a or self.all_in: 

    
            self.SE_embeddings_a = (  #for start and end Audio   #only start so 1
                nn.Embedding(1, self.embedding_dim_a, padding_idx=None)
                if self.is_start_AV_embeddings
                else None
            )
        
            self.padding_idx_a=1#1  #take one when you use padding mask with the forward function

            #Max positions 310 and dimentions of 512
            self.embed_positions_a = (  #We need three postional embeddings matrix for each modality
                PositionalEmbeddingMul(
                    self.max_seq_len_a,
                    self.embedding_dim_a,
                    padding_idx=(self.padding_idx_a  if offset_positions_by_padding else None),
                    learned=self.learned_pos_embedding,
                )
                if self.use_position_embeddings
                else None
            )

       




            self.layers_a = nn.ModuleList(  #Text to Audio (The query vector comes from the Text and Key-Value from the Audio)
                [
                    TransformerSentenceEncoderLayer(
                        embedding_dim=self.embedding_dim_a,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        add_bias_kv=add_bias_kv,
                        add_zero_attn=add_zero_attn,
                        export=export,

                    )
                    for _ in range(num_encoder_layers)
                ]
                if is_self_attention
                else None
            )

      



        if (self.all_in) or (self.only_a and self.only_t):

        
            self.layers_ta = nn.ModuleList(  #Text to Audio (The query vector comes from the Text and Key-Value from the Audio)
                [
                    TransformerMultiEncoderLayer(
                        embedding_dim=self.embedding_dim_t,#self.embedding_dim,
                        qdim=self.embedding_dim_t,
                        kdim=self.embedding_dim_a,
                        vdim=self.embedding_dim_a,
                        self_attention=False,
                        encoder_decoder_attention=True,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        add_bias_kv=add_bias_kv,
                        add_zero_attn=add_zero_attn,
                        export=export,
                    )
                    for _ in range(num_encoder_layers_cross)
                ]
            )

            self.layers_at = nn.ModuleList( #Audio to Text  (The query vector comes from the Audio and Key-Value from the Text)
                [
                    TransformerMultiEncoderLayer(
                        embedding_dim=self.embedding_dim_a,#self.embedding_dim,
                        qdim=self.embedding_dim_a,
                        kdim=self.embedding_dim_t,
                        vdim=self.embedding_dim_t,
                        self_attention=False,
                        encoder_decoder_attention=True,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        add_bias_kv=add_bias_kv,
                        add_zero_attn=add_zero_attn,
                        export=export,
                    )
                    for _ in range(num_encoder_layers_cross)
                ]
            )

 
        if (self.all_in) or (self.only_a and self.only_v):

            self.layers_av = nn.ModuleList( #Audio to Video  (The query vector comes from the Audio and Key-Value from the Video)
                [
                    TransformerMultiEncoderLayer(
                        embedding_dim=self.embedding_dim_a,#self.embedding_dim,
                        qdim=self.embedding_dim_a,
                        kdim=self.embedding_dim_v,
                        vdim=self.embedding_dim_v,
                        self_attention=False,
                        encoder_decoder_attention=True,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        add_bias_kv=add_bias_kv,
                        add_zero_attn=add_zero_attn,
                        export=export,
                    )
                    for _ in range(num_encoder_layers_cross)
                ]
            )


            self.layers_va = nn.ModuleList(  #Video to Audio (The query vector comes from the Video and Key-Value from the Audio)
                [
                    TransformerMultiEncoderLayer(
                        embedding_dim=self.embedding_dim_v,#self.embedding_dim,
                        qdim=self.embedding_dim_v,
                        kdim=self.embedding_dim_a,
                        vdim=self.embedding_dim_a,
                        self_attention=False,
                        encoder_decoder_attention=True,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        add_bias_kv=add_bias_kv,
                        add_zero_attn=add_zero_attn,
                        export=export,
                    )
                    for _ in range(num_encoder_layers_cross)
                ]
            )


        if (self.all_in) or (self.only_t and self.only_v):
            self.layers_vt = nn.ModuleList( #Video to Text  (The query vector comes from the Video and Key-Value from the Text)
                [
                    TransformerMultiEncoderLayer(
                        embedding_dim=self.embedding_dim_v,#self.embedding_dim,
                        qdim=self.embedding_dim_v,
                        kdim=self.embedding_dim_t,
                        vdim=self.embedding_dim_t,
                        self_attention=False,
                        encoder_decoder_attention=True,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        add_bias_kv=add_bias_kv,
                        add_zero_attn=add_zero_attn,
                        export=export,
                    )
                    for _ in range(num_encoder_layers_cross)
                ]
            )

            self.layers_tv = nn.ModuleList( #Text to Video  (The query vector comes from the Text and Key-Value from the video)
                [
                    TransformerMultiEncoderLayer(
                        embedding_dim=self.embedding_dim_t,#self.embedding_dim,
                        qdim=self.embedding_dim_t,
                        kdim=self.embedding_dim_v,
                        vdim=self.embedding_dim_v,
                        self_attention=False,
                        encoder_decoder_attention=True,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        add_bias_kv=add_bias_kv,
                        add_zero_attn=add_zero_attn,
                        export=export,
                    )
                    for _ in range(num_encoder_layers_cross)
                ]
            )


  
        if encoder_normalize_before: 

            
            if self.only_a or self.all_in: 
                self.emb_layer_norm_a = LayerNorm(self.embedding_dim_a, export=export)

            else:
                self.emb_layer_norm_a = None

            if self.only_v or self.all_in: 
                self.emb_layer_norm_v = LayerNorm(self.embedding_dim_v, export=export)

            else:
                self.emb_layer_norm_v = None

        else:
            self.emb_layer_norm_a = None
            self.emb_layer_norm_v = None

        

     
        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False
        
        if freeze_embeddings:
      
            #freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)


        for layer in range(n_trans_layers_to_freeze): #Can freeze first few layers with this way 
            freeze_module_params(self.layers[layer])

 
     

    def forward(
        self,
        multi_modal_features: dict, #This tensor consist of output vectors from the three modalities 
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


       
        text_features=multi_modal_features['Text']
        audio_features=multi_modal_features['Audio_c']
        padded_audio_samples=multi_modal_features['padded_audio']
        video_features=multi_modal_features['Video']

        multi_modal_tokens=multi_modal_features['raw_data']

        ############# These can be used to calculate pad mask #######
        raw_tokens_text=multi_modal_tokens['text']
        raw_tokens_audio=multi_modal_tokens['audio']
        raw_tokens_video=multi_modal_tokens['video']
        ###############################################

     
        #It will be still hard to use pad for the audio since we cannot match the output embeddings of wav2vec to descrete input
        #For the video we can mask out easily by adding padded frames

        ########### Calculating the time steps only for the unpadded seqeunce######

        cnn_layers=[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)]

        n_in = 50000 #real size of waveform (wave length - difference)
        p = 0
        padded_audio_output_sizes=[]
        for n_in in padded_audio_samples:
            for (d,k,s) in cnn_layers:
                n_out = math.floor( ((n_in+2*p-k) / s ) +1 )
                n_in = n_out
            padded_audio_output_sizes.append(n_out)
            
        
        pad_amount_aud_tensor=torch.from_numpy(np.array(padded_audio_output_sizes)).unsqueeze(1)
        padding_mask_text = raw_tokens_text.eq(1)  #Getting a mask for attention , 0s for elements with actual value
        padding_mask_audio = raw_tokens_audio.eq(-2)
        padding_mask_video =raw_tokens_video[:,0,:,0,0].eq(-1)

        padding_mask_audio=torch.ones(pad_amount_aud_tensor.shape[0],audio_features.shape[1]).type_as(padding_mask_text)
        num_zeros = audio_features.shape[1] - pad_amount_aud_tensor

        for i in range(audio_features.shape[0]):
            padding_mask_audio[i,:num_zeros[i]] = 0



        if self.is_start_AV_embeddings: #Adding the new two tokens to the dataset 

            #print(padding_mask_audio,"sssss")

            #CHANGE
            element=torch.zeros(padding_mask_video.shape[0],1).type_as(padding_mask_video)
            

            if self.only_a or self.all_in:
                padding_mask_audio=torch.cat((element,padding_mask_audio),dim=1)
            #We are not using stop token
            #padding_mask_audio=torch.cat((padding_mask_audio,element),dim=1) #padding zero to the stop token audio
            
            if self.only_v or self.all_in: 

                padding_mask_video=torch.cat((element,padding_mask_video),dim=1)
            #We are not using stop token
            #padding_mask_video=torch.cat((padding_mask_video,element),dim=1)  #padding zero to the stop token video

        

        if self.only_t or self.all_in:
            if not padding_mask_text.any():
                padding_mask_text = None
            x_t =text_features
            if self.embed_scale is not None:
                x_t *= self.embed_scale
                


        if self.only_a or self.all_in:
            padding_mask_audio_p = padding_mask_audio
            if not padding_mask_audio.any():
                padding_mask_audio = None
            x_a =audio_features
            if self.embed_scale is not None:
                x_a *= self.embed_scale

        if self.only_v or self.all_in:
            padding_mask_video_p = padding_mask_video   
            if not padding_mask_video.any():
                padding_mask_video = None 
            x_v =video_features
            if self.embed_scale is not None:
                x_t *= self.embed_scale

       
        
        # padding_mask_video_p = padding_mask_video
        # padding_mask_audio_p = padding_mask_audio
    
        #No padding Because we send embeddings not raw text
        # if not padding_mask_video.any():
        #     padding_mask_video = None

        # if not padding_mask_text.any():
        #     padding_mask_text = None
        

        # if not padding_mask_audio.any():
        #     padding_mask_audio = None

        

        # ########### Real Data##################
        # x_t =text_features
        # x_a =audio_features
        # x_v =video_features
        # #######################################
        

        # if self.embed_scale is not None:
        #     x_t *= self.embed_scale
        #     x_a *= self.embed_scale
        #     x_v *= self.embed_scale
         
        
        if self.only_a or self.all_in:
            if self.is_start_AV_embeddings: #for audio
                start_av=torch.cuda.LongTensor(video_features.shape[0], 1).fill_(0)#torch.zeros(video_features.shape[0],1)
                #end_av=torch.cuda.LongTensor(video_features.shape[0], 1).fill_(1)#torch.ones(audio_features.shape[0],1)

                start_a=self.SE_embeddings_a(start_av)
                #end_a=self.SE_embeddings_a(end_av)

            
                x_a=torch.cat((start_a,x_a),dim=1)
                #x_a=torch.cat((x_a,end_a),dim=1) #for the end token

          
        
       
        if self.only_v or self.all_in:
            if self.is_start_AV_embeddings: # for video
                start_av=torch.cuda.LongTensor(video_features.shape[0], 1).fill_(0)
                start_v=self.SE_embeddings_v(start_av)
                #end_v=self.SE_embeddings_v(end_av)

                x_v=torch.cat((start_v,x_v),dim=1)
                #x_v=torch.cat((x_v,end_v),dim=1) #for the end token

    

        if self.only_a or self.all_in:
        ############### We do Not Use################################################
            if self.embed_positions_a is not None:
                #x_a += self.embed_positions_a(x_a[:,:,0], positions=positions)
                x_a += self.embed_positions_a(padding_mask_audio_p, positions=positions)
            

        if self.only_v or self.all_in:            
        
            if self.embed_positions_v is not None:
                #x_v += self.embed_positions_v(x_v[:,:,0], positions=positions)
                x_v += self.embed_positions_v(padding_mask_video_p, positions=positions)
            
      
        # print(self.segment_embeddings)

        
        # if self.segment_embeddings is not None and segment_labels is not None:
        #     print("I have segment embeddings")
        #     x_t += self.segment_embeddings(segment_labels)
        #     x_a += self.segment_embeddings(segment_labels)
        #     x_v += self.segment_embeddings(segment_labels)

   
        ################################################################################


        if self.emb_layer_norm_a is not None: 
            x_a = self.emb_layer_norm_a(x_a)
           
       
        if self.emb_layer_norm_v is not None:  
            x_v = self.emb_layer_norm_v(x_v)


        last_states={}
        seq_rep={}

        # seq_rep = {
        #     't2a_r': ta_rep,
        #     'a2t_r': at_rep,
        #     'a2v_r': av_rep,
        #     'v2a_r': va_rep,
        #     't2v_r': tv_rep,
        #     'v2t_r': vt_rep,
    



        if self.only_a or self.all_in: 
            x_a = F.dropout(x_a, p=self.dropout, training=self.training)
            if padding_mask_audio is not None:
                x_a *= 1 - padding_mask_audio.unsqueeze(-1).type_as(x_a)

            x_a = x_a.transpose(0, 1)

            if self.is_start_AV_embeddings:
                for layer_a in self.layers_a:  #mask should be the key
                    x_a,_=layer_a(x_a,self_attn_padding_mask=padding_mask_audio)

            
            j_aud_n=x_a[0, :, :]

            seq_rep.update({'j_aud' : j_aud_n})
            

           


        if self.only_v or self.all_in:
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)


            if padding_mask_video is not None:
                x_v *= 1 - padding_mask_video.unsqueeze(-1).type_as(x_v)

            x_v = x_v.transpose(0, 1)

            if self.is_start_AV_embeddings:
                for layer_v in self.layers_v:  #mask should be the key
                    x_v,_=layer_v(x_v,self_attn_padding_mask=padding_mask_video)


            j_vid_n=x_v[0, :, :]
            seq_rep.update({'j_vid' : j_vid_n})


        # # account for padding while computing the representation
        # if padding_mask_audio is not None:
        #     x_a *= 1 - padding_mask_audio.unsqueeze(-1).type_as(x_a)
           
            
        # if padding_mask_video is not None:
        #     x_v *= 1 - padding_mask_video.unsqueeze(-1).type_as(x_v)

        if self.only_t or self.all_in:

            if padding_mask_text is not None:
                x_t *= 1 - padding_mask_text.unsqueeze(-1).type_as(x_t)

        
            j_text=x_t[:, 0, :] #text embeddigs

            seq_rep.update({'j_text' : j_text})

            x_t = x_t.transpose(0, 1)
        
        # # B x T x C -> T x B x C
        # x_t = x_t.transpose(0, 1)
        # x_a = x_a.transpose(0, 1)
        # x_v = x_v.transpose(0, 1)


 

    
        # if self.is_start_AV_embeddings:
        #     for layer_a in self.layers_a:  #mask should be the key
        #         x_a,_=layer_a(x_a,self_attn_padding_mask=padding_mask_audio)

        
        # j_aud_n=x_a[0, :, :]

     

       
        # if self.is_start_AV_embeddings:
        #     for layer_v in self.layers_v:  #mask should be the key
        #         x_v,_=layer_v(x_v,self_attn_padding_mask=padding_mask_video)


        # j_vid_n=x_v[0, :, :]


        if (self.all_in) or (self.only_t and self.only_a):
            x_ta=x_t
            for layer_ta in self.layers_ta:  #mask should be the key
                x_ta,_=layer_ta(x_ta,x_a,x_a, self_attn_padding_mask=padding_mask_audio)
                

            x_at=x_a
            for layer_at in self.layers_at:
                x_at,_=layer_at(x_at,x_t,x_t, self_attn_padding_mask=padding_mask_text)

            x_ta = x_ta.transpose(0, 1)
            x_at = x_at.transpose(0, 1)

            ta_rep = x_ta[:, 0, :]
            at_rep = x_at[:, 0, :]

            seq_rep.update({'t2a_r' : ta_rep})
            seq_rep.update({'a2t_r' : at_rep})
         

        if (self.all_in) or (self.only_v and self.only_a):

            x_av=x_a
            for layer_av in self.layers_av:
                x_av,_=layer_av(x_av,x_v,x_v, self_attn_padding_mask=padding_mask_video)

            x_va=x_v
            for layer_va in self.layers_va:
                x_va,_=layer_va(x_va,x_a,x_a, self_attn_padding_mask=padding_mask_audio)

            x_av = x_av.transpose(0, 1)
            x_va = x_va.transpose(0, 1)

            av_rep = x_av[:, 0, :]
            va_rep = x_va[:, 0, :]

            seq_rep.update({'a2v_r' : av_rep})
            seq_rep.update({'v2a_r' : va_rep})


        if (self.all_in) or (self.only_v and self.only_t):        

            x_tv=x_t
            for layer_tv in self.layers_tv:
                x_tv,_=layer_tv(x_tv,x_v,x_v, self_attn_padding_mask=padding_mask_video)

            x_vt=x_v
            for layer_vt in self.layers_vt:
                x_vt,_=layer_vt(x_vt,x_t,x_t, self_attn_padding_mask=padding_mask_text)

            x_tv = x_tv.transpose(0, 1)
            x_vt = x_vt.transpose(0, 1) 

            tv_rep = x_tv[:, 0, :]
            vt_rep = x_vt[:, 0, :]    

            seq_rep.update({'t2v_r' : tv_rep})
            seq_rep.update({'v2t_r' : vt_rep})

            

        # T x B x C -> B x T x C
        # x_ta = x_ta.transpose(0, 1)
        # x_at = x_at.transpose(0, 1)
        # x_av = x_av.transpose(0, 1)
        # x_va = x_va.transpose(0, 1)
        # x_tv = x_tv.transpose(0, 1)
        # x_vt = x_vt.transpose(0, 1) 

     


        # sentence_rep = x[:, 0, :]
        # ta_rep = x_ta[:, 0, :]
        # at_rep = x_at[:, 0, :]
        # av_rep = x_av[:, 0, :]
        # va_rep = x_va[:, 0, :]
        # tv_rep = x_tv[:, 0, :]
        # vt_rep = x_vt[:, 0, :]

    
        # last_states = {
        #     't2a': x_ta,
        #     'a2t': x_at,
        #     'a2v': x_av,
        #     'v2a': x_va,
        #     't2v': x_tv,
        #     'v2t': x_vt
        # }

        # seq_rep = {
        #     't2a_r': ta_rep,
        #     'a2t_r': at_rep,
        #     'a2v_r': av_rep,
        #     'v2a_r': va_rep,
        #     't2v_r': tv_rep,
        #     'v2t_r': vt_rep,
        #     'j_text':j_next,
        #     'j_aud' :j_aud_n,
        #     'j_vid' :j_vid_n
        # }

       
        return last_states, seq_rep
