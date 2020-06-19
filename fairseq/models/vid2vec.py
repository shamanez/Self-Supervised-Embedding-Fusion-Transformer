# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from . import (
#     BaseFairseqModel, register_model, register_model_architecture
# )


# @register_model('vid2vec')
# class Vid2VecModel(BaseFairseqModel):

#     @staticmethod
#     def add_args(parser):
#         """Add model-specific arguments to the parser."""
#         parser.add_argument('--output_size_v', type=int, metavar='N', help='number of output features')
#         parser.add_argument('--num_filters_v', type=int, metavar='N',
#                             help='initial number of output filteres')

#     @classmethod
#     def build_model(cls, args, task):
#         """Build a new model instance."""
       
#         # make sure all arguments are present in older models
#         base_vid2vec_architecture(args)

#         model = Vid2VecModel(args)
#         print(model)
#         return model

#     def __init__(self, args):
#         super().__init__()
#         self.feature_extractor=ConvFeatureExtractionModel(self, output_size=args.output_size_v, num_filters=args.num_filters_v)



#     def forward(self, source):
#         result = {}

#         features = self.feature_extractor(source)
#         x = self.dropout_feats(features)
   
#         result['features'] = x
     
#         return result

#     def upgrade_state_dict_named(self, state_dict, name):
#         return state_dict



# class ConvFeatureExtractionModel(nn.Module):
#     def __init__(self,output_size=128, num_filters=64):
#         super().__init__()

#         conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
#         conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
#         conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
#         conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
#         conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
#         conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
#         conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
#         conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

#         batch_norm = nn.BatchNorm2d(num_filters)
#         batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
#         batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
#         batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
#         batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
#         batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
#         batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

#         leaky_relu = nn.LeakyReLU(0.2, True)

#         self.encoder=nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
#                               leaky_relu, conv3, batch_norm4_0, leaky_relu, \
#                               conv4, batch_norm8_0, leaky_relu, conv5, 
#                               batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
#                               leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)

#     def forward(self, x):
#         # BxT -> BxCxT
#         #x = x.unsqueeze(1)
#         x=self.encoder(x)
#         return x





# # @register_model_architecture('vid2vec', 'vid2vec')
# # def base_vid2vec_architecture(args):

# #     args.output_size_v = getattr(args, 'output_size_v', 256)
# #     args.num_filters_v = getattr(args, 'num_filters_v', 32)
 


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, input_nc, num_decoders=5, inner_nc=128, num_additional_ids=0, smaller=False, num_masks=0):
        super(ConvFeatureExtractionModel, self).__init__()

        self.encoder = self.generate_encoder_layers(output_size=inner_nc, num_filters=num_additional_ids)
        # self.decoder = self.generate_decoder_layers(inner_nc*2, num_filters=num_additional_ids)
        # self.mask = self.generate_decoder_layers(inner_nc*2, num_output_channels=1, num_filters=num_additional_ids)


    def forward(self, x):
        x=self.encoder(x)
        return x


    def generate_encoder_layers(self, output_size=128, num_filters=64):
        conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
        conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
        conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
        conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
        conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

        batch_norm = nn.BatchNorm2d(num_filters)
        batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
        batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
        batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

        leaky_relu = nn.LeakyReLU(0.2, True)
        return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                        leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                        conv4, batch_norm8_0, leaky_relu, conv5, 
                        batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                        leaky_relu, conv7, batch_norm8_3, leaky_relu, conv8)








	# def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=32):
	# 	up = nn.Upsample(scale_factor=2, mode='bilinear')

	# 	dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
	# 	dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
	# 	dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
	# 	dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
	# 	dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
	# 	dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
	# 	dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
	# 	dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

	# 	batch_norm = nn.BatchNorm2d(num_filters)
	# 	batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
	# 	batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
	# 	batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
	# 	batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
	# 	batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

	# 	leaky_relu = nn.LeakyReLU(0.2)
	# 	relu = nn.ReLU()
	# 	tanh = nn.Tanh()

	# 	return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4, \
	# 						relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5, relu,
	# 						nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4,
	# 						batch_norm8_7, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1, 
	# 						relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1, relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
	# 						relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8, tanh)
		


	# def generate_decoder_small_layers(self, num_input_channels, num_output_channels=2, num_filters=16):
	# 	up = nn.Upsample(scale_factor=2, mode='bilinear')

	# 	dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
	# 	dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
	# 	dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
	# 	dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
	# 	dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
	# 	dconv6 = nn.Conv2d(num_filters * 2 , num_output_channels, 3, 1, 1)

	# 	batch_norm = nn.BatchNorm2d(num_filters)
	# 	batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
	# 	batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
	# 	batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
	# 	batch_norm4_1 = nn.BatchNorm2d(num_filters * 2)
	# 	batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
	# 	batch_norm8_7 = nn.BatchNorm2d(num_filters * 4)

	# 	leaky_relu = nn.LeakyReLU(0.2)
	# 	relu = nn.ReLU()
	# 	tanh = nn.Tanh()

	# 	return nn.Sequential(relu, up, dconv1, batch_norm8_4, \
	# 						relu, up, dconv2, batch_norm8_5, relu,
	# 						up, dconv3, batch_norm8_6, relu, up, dconv4,
	# 						batch_norm8_7, relu, up, dconv5, batch_norm4_1, 
	# 						relu, up, dconv6, tanh)