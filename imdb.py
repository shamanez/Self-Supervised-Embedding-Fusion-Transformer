'''
IMDB data has one data-sample in each file, below python code-snippet converts it one file for train and valid each for ease of processing.
'''
import argparse
import os
import random
from glob import glob

random.seed(0)

def main(args):
    for split in ['train', 'test']:
        samples = []
        for class_label in ['pos', 'neg']:
            fnames = glob(os.path.join(args.datadir, split, class_label) + '/*.txt')
            for fname in fnames:
                with open(fname) as fin:
                    line = fin.readline()
                    samples.append((line, 1 if class_label == 'pos' else 0))
        random.shuffle(samples)
        out_fname = 'train' if split == 'train' else 'dev'
        f1 = open(os.path.join(args.datadir, out_fname + '.input0'), 'w')
        f2 = open(os.path.join(args.datadir, out_fname + '.label'), 'w')
        for sample in samples:
            f1.write(sample[0] + '\n')
            f2.write(str(sample[1]) + '\n')
        f1.close()
        f2.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='aclImdb')
    args = parser.parse_args()
    main(args)