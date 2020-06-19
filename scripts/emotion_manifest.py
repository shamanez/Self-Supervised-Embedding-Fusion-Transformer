#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import soundfile
import random


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', metavar='DIR', help='root directory containing flac files to index')
    parser.add_argument('--valid-percent', default=0.1, type=float, metavar='D',
                        help='percentage of data to use as validation set (between 0 and 1)')
    parser.add_argument('--dest', default='.', type=str, metavar='DIR', help='output directory')
    parser.add_argument('--ext', default='pt', type=str, metavar='EXT', help='extension to look for')
    parser.add_argument('--seed', default=42, type=int, metavar='N', help='random seed')
    parser.add_argument('--path-must-contain', default=None, type=str, metavar='FRAG',
                        help='if set, path must contain this substring for a file to be included in the manifest')
    return parser

#python emotion_data_preprocessing.py -a '/media/gsir059/Transcend/Rivindu/raw_data/Audio/' -t '/media/gsir059/Transcend/Rivindu/raw_data/Text/'

def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.

    audio_folder = 'Audio'
    video_folder = 'FaceVideo'
    text_folder = 'Text'

    dir_path_video = os.path.realpath(os.path.join(args.root,video_folder))
    dir_path_audio = os.path.realpath(os.path.join(args.root,audio_folder))
    dir_path_text = os.path.realpath(os.path.join(args.root,text_folder))
    search_path = os.path.join(dir_path_video, '**/*.mp4') #TODO: using video path as ground truth
    rand = random.Random(args.seed)

    label_map = {
        'neu': 'neutral',
        'fru': 'frustrated',
        'ang': 'angry',
        'hap': 'happy',
        'sad': 'sad',
        'sur': 'surprised',
        'exc': 'excited'
    }

    

    with open(os.path.join(args.dest, 'train_t.tsv'), 'w') as train_ft, \
         open(os.path.join(args.dest, 'valid_t.tsv'), 'w') as valid_ft, \
         open(os.path.join(args.dest, 'train_v.tsv'), 'w') as train_fv, \
         open(os.path.join(args.dest, 'valid_v.tsv'), 'w') as valid_fv, \
         open(os.path.join(args.dest, 'train_a.tsv'), 'w') as train_fa, \
         open(os.path.join(args.dest, 'valid_a.tsv'), 'w') as valid_fa, \
         open(os.path.join(args.dest, 'label_file.csv'), 'w') as label_file:
        print(dir_path_text, file=train_ft)
        print(dir_path_text, file=valid_ft)
        print(dir_path_video, file=train_fv)
        print(dir_path_video, file=valid_fv)
        print(dir_path_audio, file=train_fa)
        print(dir_path_audio, file=valid_fa)
        print('FileName,Emotion', file=label_file)

        for fname in glob.iglob(search_path, recursive=True):
            file_path_video = os.path.realpath(fname).replace('.pt','.mp4')
            file_path_audio = file_path_video.replace(video_folder,audio_folder).replace('.mp4','.pt') # Hacky
            file_path_txt = os.path.realpath(fname.replace(video_folder,text_folder).replace('.mp4','.txt'))# Hacky

            filename = os.path.basename(file_path_video).split('.')[0]
            label_str = filename.split('_')[1]
            label = label_map.get(label_str)
            print('{},{}'.format(filename,label), file=label_file)

            if args.path_must_contain and args.path_must_contain not in file_path_video:
                continue

            #frames = soundfile.info(fname).frames
            dest_v,dest_t,dest_a = (train_fv,train_ft,train_fa) if rand.random() > args.valid_percent else (valid_fv, valid_ft, valid_fa)
            print('{}\t{}'.format(os.path.relpath(file_path_video, dir_path_video), 1e6), file=dest_v)
            print('{}\t{}'.format(os.path.relpath(file_path_txt, dir_path_text), 1e6), file=dest_t)
            print('{}\t{}'.format(os.path.relpath(file_path_audio, dir_path_audio), 1e6), file=dest_a)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
