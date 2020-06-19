import os
import sys
import numpy as np

import torch
import torch.nn.functional as F

import cv2
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, Compose, ToTensor

import soundfile as sf

import glob

import argparse

# FPS=24
# Sec=7.5

# Frame =200

problem_aud=open('PROBLEM_AUD.text', 'w')

class EmotionDataPreprocessing():
    
    def __init__(self,channels=3,timeDepth=200,xSize=256,ySize=256,sample_rate=16000):
        #VIDEO
        self.channels = channels
        self.timeDepth = timeDepth #Number of frames to get
        self.xSize = xSize
        self.ySize = ySize
        IMAGE_SIZE=(self.xSize,self.ySize)
        self.transform = Compose([Resize(IMAGE_SIZE), ToTensor()])

        #TEXT
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')  #maybe we can move the roberta to forward function

        #AUDIO
        self.sample_rate = sample_rate
        
    def preprocess_video_file(self,filename):
        print("Video: ",filename)
        cap = cv2.VideoCapture(filename)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        #frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)

        frames = -1*torch.ones([self.channels, self.timeDepth, self.xSize, self.ySize], dtype=torch.float32)

        failed_clip = False
        frame_counter = 0
        for f in range(self.timeDepth):
            # repreat video if early finish
            # if frame_counter >= n_frames:
            #     frame_counter = 0
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            #     print("repeating",end='.')
            ret, frame = cap.read()
            #cv2.imshow('IMG_me',frame)
            #frame2=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # rgb_frame= frame[:, :, ::-1]
            # cv2.imshow('IMG_me2',frame2)
            # cv2.imshow('IMG_me3',rgb_frame)
            # cv2.waitKey()
            frame_counter += 1
            if ret:
                rgb_frame= frame[:, :, ::-1]
                #frame = Image.fromarray(frame, 'RGB')
                frame = Image.fromarray(rgb_frame, 'RGB')
                frame = self.transform(frame)
                frames[:, f, :, :] = frame
            elif frame_counter >= n_frames:
                break
            else:
                print("Skipped!")
                failed_clip = True
                fail_path = os.path.join('failed_video_clips',os.path.basename(filename)+'.txt')
                os.makedirs(os.path.dirname(fail_path), exist_ok=True)
                with open(fail_path, 'w') as f:
                    f.writelines(str(frame_counter)+'\t'+str(ret)+'\t'+filename)
                break
        print(": ",frame_counter)
        #import pdb; pdb.set_trace();
        return frames, failed_clip

    def preprocess_text_file(self,filename):
        #Text data
        with open(filename, 'rt') as f:
            words = []
            for line in f:
                words.extend(self.roberta.encode(line))#.split('\t')) #split by spaces
        #words='Mark told Pete many lies about himself, which Pete included in his book. He should have been more skeptical.'
        tokensized_text = [word.item() for word in words]
        print("Text: ", len(tokensized_text))
        # print(tokensized_text)
        # exit()
        return tokensized_text

    def resample(self, x, factor):
        return F.interpolate(x.view(1, 1, -1), scale_factor=factor).squeeze()

    def preprocess_audio_file(self,filename):
        #Wave data
        wav, curr_sample_rate = sf.read(filename)
        print(curr_sample_rate,filename)
        feats_audio = torch.from_numpy(wav).float()
        if feats_audio.dim() == 2:
            feats_audio = feats_audio.mean(-1)
        if curr_sample_rate != self.sample_rate:
            factor = self.sample_rate / curr_sample_rate

            feats_audio = self.resample(feats_audio, factor)
            # try:
            #     feats_audio = self.resample(feats_audio, factor)
            # except RuntimeError:
            #     print(filename)
            #     #problem_aud.writelines(filename+'\n')
            #     #exit()

        assert feats_audio.dim() == 1, feats_audio.dim()
        print("Audio: ",feats_audio.size())
        return feats_audio

    def preprocess_data(self , video_path, audio_path, text_path):
        num_items = 1e18
        current_num = 0
        #TEXT
        if text_path:
            #all_text_tokens = []
            text_files = sorted(glob.glob(text_path+"*.txt"))
            print(len(text_files)," text files found")
            for text_file in text_files:
                tokensized_text = self.preprocess_text_file(text_file)
                #all_text_tokens.append(tokensized_text)
                output_file = text_file.replace('text','text')#.replace('.txt','.pt')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    for item in tokensized_text:
                        f.write(str(item)+'\t')
                current_num += 1
                if current_num>num_items:
                    break
                #torch.save(tokensized_text,output_file)
        
        current_num = 0

        #VIDEOS
        if video_path:
            #all_videos = []
            video_files = sorted(glob.glob(video_path+"*.mp4"))
            print(len(video_files)," video_files found")
            for video_file in video_files:
                if not os.path.isfile(video_file.replace('raw_data','processed_data').replace('.mp4','.pt')):
                    frames, failed_clip = self.preprocess_video_file(video_file)
                else:
                    continue
                #all_videos.append(frames)
                # create output file
                output_file = video_file.replace('raw_data','processed_data').replace('.mp4','.pt')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                torch.save(frames,output_file)
                current_num += 1
                if current_num>num_items:
                    break
        
        current_num = 0

        #AUDIO
       

        if audio_path:
            #all_audio_features = []
            audio_files = sorted(glob.glob(audio_path+"*.wav"))
            print(len(audio_files)," audio_files found")
            for audio_file in audio_files:
                audio_features = self.preprocess_audio_file(audio_file)
                #all_audio_features.append(audio_features)
                output_file = audio_file.replace('audio','audio_pt').replace('.wav','.pt')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                torch.save(audio_features,output_file)
                current_num += 1
                if current_num>num_items:
                    break


if __name__ == "__main__":
    data_processor = EmotionDataPreprocessing()

    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--video_path', default=None, help='path for raw video files')
    parser.add_argument('-a','--audio_path', default=None, help='path for raw audio files')
    parser.add_argument('-t','--text_path', default=None, help='path for raw text files')

    args = parser.parse_args()

    video_path = args.video_path
    audio_path = args.audio_path
    text_path = args.text_path

    # python emotion_data_preprocessing.py -v '/home/1TB/Emocap-Data/raw_data/FaceVideo/' -a '/home/1TB/Emocap-Data/raw_data/Audio/' -t '/home/1TB/Emocap-Data/raw_data/Text/'
    # python emotion_data_preprocessing.py -v '/media/gsir059/Transcend/Rivindu/raw_data/FaceVideo/' -a '/media/gsir059/Transcend/Rivindu/raw_data/Audio/' -t '/media/gsir059/Transcend/Rivindu/raw_data/Text/'


    
    # video_path = "/home/1TB/FriendsData/raw_data/FaceVideo/"#/home/1TB/EvanRawData/raw_data/Video_Data/'
    audio_path = '/home/gsir059/Documents/cmu-mosei_data/train/audio_new/'
    #text_path = '/home/gsir059/Documents/cmu-mosei_data/train/text/'
    
    data_processor.preprocess_data(video_path,audio_path,text_path)
