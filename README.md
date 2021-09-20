# [Multimodal Emotion Recognition with Transformer-Based Self Supervised Feature Fusion](https://ieeexplore.ieee.org/document/9206016)


![Model Overviw](https://github.com/shamanez/Self-Supervised-Embedding-Fusion-Transformer/blob/master/figure.png)

# ****Please replace the Table 6 in the paper****

[Please replace the Table 6 of the paper with this table.](https://github.com/shamanez/Self-Supervised-Embedding-Fusion-Transformer/blob/master/MOSI.pdf)

# Basic strucutre of the code

## Inspiration from fairseq

1. This code strcuture is built on top of Faiseq interface
2. Fairseq is an open source project by FacebookAI team that combined different SOTA architectures for sequencial data processing
3. This also consist of SOTA optimizing mechanisms such as ealry stopage, warup learnign rates, learning rate shedulers
4. We are trying to develop our own architecture in compatible with fairseq interface. 
5. For more understanding please read the [paper](https://arxiv.org/abs/1904.01038) published about Fairseq interaface.

## Merging of our own architecture with Fairseq interface

1. This can be bit tricky in the beggining. First  it is important to udnestand that Fairseq has built in a way that all architectures can be access through the terminal commands (args).

2. Since our architecture has lot of properties in tranformer architecture, we followed the [a tutorial](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.custom_classification.md) that describe to use Roberta for the custom classification task.

3. We build over archtiecture by inserting new stuff to following directories in Fairseq interfeace.
   - fairseq/data
   - fairseq/models
   - fairseq/modules
   - fairseq/tasks
   - fairseq/criterions


# Main scripts of the code

## Our main scripts are categorized in to for parts

1. Custom dataloader for load raw audio, faceframes and text is in the **fairseq/data/raw_audio_text_video_dataset.py**

2. The task of the emotion prediction similar to other tasks such as translation is in the **fairseq/tasks/emotion_prediction.py**

3. The custom architecture of our model similar to roberta,wav2vec is in the **fairseq/models/mulT_emo.py**

4. To obtain Inter-Modal attention we modify the self attentional architecture a bit. They can be found in **fairseq/modules/transformer_multi_encoder.py** and  **fairseq/modules/transformer_layer.py**

5. Finally the cutom loss function scripts  cab be found it **fairseq/criterions/emotion_prediction_cri.py**



# Prerequest models 

### Our model uses pretrained SSL methods to extract features. It is important to download those checkpoints prior to the trainig procedure. Please you the following links to downlaod the pretrained SSL models.

1. For audio fetures - [wav2vec](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) 
2. For facial features - [Fabnet](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/fabnet.html)
3. For sentence (text) features - [Roberta](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)



## Training Command


python train.py  --data ./T_data-old/mosei_sent    --restore-file None   --task emotion_prediction     --reset-optimizer --reset-dataloader --reset-meters     --init-token 0 --separator-token 2     --arch robertEMO_large     --criterion emotion_prediction_cri     --num-classes 1     --dropout 0.1 --attention-dropout 0.1     --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06     --clip-norm 0.0  --lr 1e-03  --max-epoch 32     --best-checkpoint-metric loss     --encoder-layers 2  --encoder-attention-heads 4 --max-sample-size 150000  --max-tokens 150000000 --batch-size 4  --encoder-layers-cross 2  --max-positions-t 512  --max-positions-a 936 --max-positions-v 301  --no-epoch-checkpoints   --update-freq 2 --find-unused-parameters --ddp-backend=no_c10d --lr-scheduler reduce_lr_on_plateau --regression-target-mos


## Validation Command


CUDA_VISIBLE_DEVICES=1 python validate.py  --data ./T_data/emocap   --path './checkpoints/checkpoint_best.pt' --task emotion_prediction --valid-subset test --batch-size 4
