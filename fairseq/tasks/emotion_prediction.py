# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np

from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset,
    RawAudioDataset,
    RawAudioTextVideoDataset
)

from . import FairseqTask, register_task


@register_task('emotion_prediction')
class EmotionPredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--data', metavar='FILE',
                            help='file prefix for Text data')


        parser.add_argument('--sample-rate', default=16000, type=int,
                            help='target sample rate. audio files will be up/down sampled to this rate')

        parser.add_argument('--max-sample-size', default=None, type=int,
                            help='max sample size to crop to for batching. default = min sample length')
        parser.add_argument('--min-sample-size', default=None, type=int,
                            help='min sample size to crop to for batching. default = same as --max-sample-size')

        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='Truncate sequence to max_sequence_length')

        ### cutomizing acording to the dataset
        parser.add_argument('--regression-target-mos', action='store_true', default=False)
        parser.add_argument('--binary-target-iemocap', action='store_true', default=False)
        parser.add_argument('--softmax-target-meld', action='store_true', default=False)
        parser.add_argument('--softmax-target-binary-meld', action='store_true', default=False)

        parser.add_argument('--eval-matric', action='store_true', default=False)  # This is to get the paper evaluations for each dataset
                            

    def __init__(self, args):
        super().__init__(args)


    @classmethod
    def setup_task(cls, args, **kwargs): # to_d0 :Might need to modify this also to merge audio and video
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)  #Initalization of an instance of our class

    def load_dataset(self, split, combine=False, **kwargs): 

     
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """   

        
       
        self.datasets[split] = RawAudioTextVideoDataset(base_path=self.args.data,
                                               data_args=self.args, 
                                               data_split=split, 
                                               sample_rate=self.args.sample_rate,
                                               max_sample_size=self.args.max_sample_size,
                                               min_sample_size=self.args.min_sample_size)

    def build_model(self, args):
        
        from fairseq import models
      
        model = models.build_model(args, self) #/home/gsir059/Documents/PhD/MulFie/fairseq/fairseq/models/__init__.py'
        print("Model Initialization Done")
    
        model.register_classification_head(
            'emotion_classification_head',
            num_classes=self.args.num_classes,
        )
      
        return model

    # def max_positions(self):
    #     return self.args.max_positions

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None