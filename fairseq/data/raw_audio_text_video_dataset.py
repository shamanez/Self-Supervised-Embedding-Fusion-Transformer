#change the line 331

import os
import numpy as np
import sys
import torch

from .import FairseqDataset

import cv2
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, Compose, ToTensor

import time

class RawAudioTextVideoDataset(FairseqDataset):

    def __init__(self, base_path,data_args,data_split, sample_rate, max_sample_size=None, min_sample_size=None,
                 shuffle=True):
        super().__init__()

        self.data_args=data_args

        self.sample_rate = sample_rate

        self.fnames_audio = []
        self.fnames_text = []
        self.fnames_video = []

        self.sizes_audio = []
        self.sizes_video = []

        self.labels = {}

        #####Video Frame #####
        self.channels = 3
        self.timeDepth = 300
        self.xSize = 256
        self.ySize = 256

        IMAGE_SIZE=(self.xSize,self.ySize)
        self.transform = Compose([Resize(IMAGE_SIZE), ToTensor()])


        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize
        self.min_sample_size = min_sample_size if min_sample_size is not None else self.max_sample_size
        self.base_manifest_path = base_path
        self.split = data_split

        if self.data_args.binary_target_iemocap: 
            included_emotions = ['neu','ang','sad','hap'] # 'exc', IEMOCAP  (Max 5 emotions (only take 4 in prior work))
        
        elif self.data_args.softmax_target_meld:

            print("We are using MELD for the softmax classification")

            included_emotions = ['neutral','sadness','surprise','joy','anger','fear','disgust'] #MELD (Max 7 emotion)
            #included_emotions = ['neutral','sadness','surprise','joy','anger']



        elif self.data_args.softmax_target_binary_meld:

            included_emotions = ['neutral','sadness','surprise','joy','anger','fear','disgust'] #MELD (Max 7 emotion)


        else:
            print("We are using MOSEI or MOSI to do a regression task")

        

        manifest_audio = os.path.join(self.base_manifest_path, '{}.tsv'.format(self.split+"_a"))
        manifest_text = os.path.join(self.base_manifest_path, '{}.tsv'.format(self.split+"_t"))
        manifest_video = os.path.join(self.base_manifest_path, '{}.tsv'.format(self.split+"_v"))
        
      
        
        manifest_label = os.path.join(self.base_manifest_path, '{}.csv'.format("label_file_"+self.split))


        with open(manifest_label, 'r') as f_l :
            self.root_dir_l = f_l.readline().strip()
            for line_l in f_l:

                items_l = line_l.strip().split(',')

                if self.data_args.regression_target_mos:                
                    self.labels[items_l[0].strip()] = np.round(float(items_l[1].strip()),decimals=4)
                else:
                    self.labels[items_l[0].strip()] = items_l[1].strip() #for the sentiment use 2 from the list else 1


        #inter_n=0
        with open(manifest_audio, 'r') as f_a, open(manifest_text, 'r') as f_t, open(manifest_video, 'r') as f_v :#, open(manifest_label, 'r') as f_l:
            self.root_dir_a = f_a.readline().strip()
            self.root_dir_t = f_t.readline().strip()
            self.root_dir_v = f_v.readline().strip()


            for line_a, line_t, line_v in zip(f_a,f_t,f_v):#,f_l):, line_l
           
                items_a = line_a.strip().split('\t')
                items_t = line_t.strip().split('\t')
                items_v = line_v.strip().split('\t')

                # inter_n=inter_n+1

                # if inter_n>5:
                #     break

                assert items_a[0].split('.')[0] == items_t[0].split('.')[0] == items_v[0].split('.')[0], "misalignment of data"
        
                emotion = self.labels.get(items_v[0].split('.')[0]) #If the label is not there, gives a none

            

                if self.data_args.regression_target_mos:

                    if self.data_args.eval_matric:
                        if emotion==0.0:
                            continue  

                    self.fnames_audio.append(items_a[0])
                    self.fnames_text.append(items_t[0])
                    self.fnames_video.append(items_v[0])
                    self.sizes_audio.append(1000000)     #This is used in the data loader np.lexsort but can remove it
                    self.sizes_video.append(1000000)
                
                else:
        
                    if emotion in included_emotions: # Only using the subset of emotions


                        self.fnames_audio.append(items_a[0])
                        self.fnames_text.append(items_t[0])
                        self.fnames_video.append(items_v[0])

                        self.sizes_audio.append(1000000)  
                        self.sizes_video.append(1000000)
                

                    
   

        if self.data_args.binary_target_iemocap:

            self.emotion_dictionary = { #EMOCAP
                'neu':0,
                'ang':2,
                'hap':3,
                'sad':1,
                #'exc':3
            }

        if self.data_args.softmax_target_meld: 

            self.emotion_dictionary = { #MELD
                'anger'  : 2,
                'joy':     3,
                'neutral': 0,
                'sadness': 1,
                'surprise':4,
                'fear':5,
                'disgust':6
            }

            # self.emotion_dictionary = { #MELD
            #     'anger'  : 2,
            #     'joy':     3,
            #     'neutral': 0,
            #     'sadness': 1,
            #     'surprise':4,
            #     #'fear':5,
            #     #'disgust':6
            # }

        if self.data_args.regression_target_mos:

            self.emotion_dictionary = {   #modei senti
                '-3'  : 6,
                '-2':     5,
                '-1': 4,
                '0': 0,
                '1':1,
                '2':2,
                '3':3
            }

            # self.emotion_dictionary = {   #modei senti 2 class
            
            #     '0': 0,
            #     '1':1
            # }
   
        self.shuffle = shuffle
    
    def preprocess_video_file(self,filename):

    
        start = time.time()
        cap = cv2.VideoCapture(filename)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        #frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)
        frames = -1*torch.ones([self.channels, self.timeDepth, self.xSize, self.ySize], dtype=torch.float32)

        failed_clip = False
        frame_counter = 0
        for f in range(self.timeDepth):
            ret, frame = cap.read()
            frame_counter += 1
            if ret:
                rgb_frame= frame[:, :, ::-1]
                frame = Image.fromarray(rgb_frame, 'RGB')
                frame = self.transform(frame)
                frames[:, f, :, :] = frame
            elif frame_counter > n_frames:
                break
            else:
                print("Skipped!")
                failed_clip = True
                break
        #print(time.time()-start,",")
        return frames, failed_clip


    def __getitem__(self, index):

    
        audio_file = self.fnames_audio[index]
        text_file = self.fnames_text[index]
        video_file = self.fnames_video[index]

        fname_a = os.path.join(self.root_dir_a, audio_file)
        fname_t = os.path.join(self.root_dir_t, text_file)
        fname_v = os.path.join(self.root_dir_v, video_file)

        file_name = audio_file.replace('.wav','')
     
        assert file_name == video_file.replace('.mp4','') and file_name == text_file.replace('.txt',''), "not all file ids match"

  

        if self.data_args.regression_target_mos:        
            label = self.labels.get(file_name)
        else:
            label = self.emotion_dictionary[str(self.labels.get(file_name))]


     
  
        # Audio data (Wav2Vec Features)
        audio_features = torch.load(fname_a.replace('.wav','.pt'))


        # Text data (Roberta Tokens)
        with open(fname_t, 'r') as f:
            words = []
            for line in f:
                words.extend(line.strip().split('\t'))
        tokensized_text = [int(word) for word in words]
        tokensized_text = torch.from_numpy(np.array(tokensized_text))
     

 
        video_frames, _ = self.preprocess_video_file(fname_v)
   
        return {
            'id': index,
            'audio': audio_features,
            'text': tokensized_text,
            'video' : video_frames,
            'target' : label,
        }

    def __len__(self): #Training dataset size
        return len(self.fnames_audio)

    def collate_tokens(self, values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        """Convert a list of 1d tensors into a padded 2d tensor."""

        size = 512#max(v.size(0) for v in values) #Here the size can be fixed as 512
        res = values[0].new(len(values), size).fill_(pad_idx)
        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)
        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res
    
    def collater(self, samples):
        
        if len(samples) == 0:
            return {}
        #collater for audio data    
        ###################################################################    
        sources = [s['audio'] for s in samples]
        sizes = [len(s) for s in sources]
        target_size = self.max_sample_size#min(min(sizes), self.max_sample_size)

        if self.min_sample_size < target_size:
            target_size = np.random.randint(self.min_sample_size, target_size + 1)

        collated_sources = sources[0].new(len(sources), target_size) #Is this for make the sizes alike
        padded_amount=[]
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            #print(diff)
            padded_amount.append(abs(diff))
            #assert diff >= 0
            if diff == 0:
                collated_sources[i] = source
            elif diff > 0:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
                collated_sources[i] = source[start:end]
            else:
                delta = abs(diff)
                #m = torch.nn.ConstantPad1d((0,delta),source[-1]) #left padding with the last element
                m = torch.nn.ConstantPad1d((0,delta),0)
                collated_sources[i] = m(source)
        
                
        ####################################################################
        #collater for text chunks        
        #############################################
        sources_text = [s['text'] for s in samples]
        collated_text = self.collate_tokens(sources_text, 1) #1 is the padding index

        ##############################################
        #collater of video chunks
        #############################################
        sources_video = [s['video'] for s in samples] #This is already fixed
        collated_video = sources_video[0].new(len(sources_video), self.channels, self.timeDepth, self.xSize, self.ySize)
        for idx_v, v_ex in enumerate(sources_video):
            collated_video[idx_v] = v_ex
        #############################################

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'net_input': {
                'audio': collated_sources,  #this was sources before
                'padded_amount':padded_amount,
                'text': collated_text, 
                'video': collated_video
            },
            #'target': torch.LongTensor([int(s['target']) for s in samples])
            'target': torch.FloatTensor([float(s['target']) for s in samples]) #onlt mosei
        }

    def get_dummy_batch(
            self, num_tokens, max_positions, src_lne=2048, tgt_len=128,
    ):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            src_len = min(src_len, max_positions)
        bsz = num_tokens // src_len
        
        return self.collater([
            {
                'id': i,
                'audio': torch.rand(self.channels, self.timeDepth, self.xSize, self.ySize),
                'text': torch.rand(src_len),
                'video' : torch.rand(src_len)
            }
            for i in range(bsz)
            ])

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self.sizes_audio[index], self.max_sample_size)

    def ordered_indices(self):  #Need to customize this
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:  #Shuffeling the training dataset
            order = [np.random.permutation(len(self))]
    
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes_audio)

        return np.lexsort(order)
