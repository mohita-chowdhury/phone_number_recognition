import torch
import os
import model
import cv2
import collections
import speech_recognition as sr 
import os 
from tqdm import tqdm
from pydub import AudioSegment 
from pydub.silence import split_on_silence 
import warnings
import argparse
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


PATH = '/home/mohiitaa/ufonia/phone_number_identification_from_speech/pretrained_model/speech_net_aug.pth.tar'
# device = torch.device("cuda")

def silence_based_conversion(path, n, sil_thresh):  
    print([path, n])
    song = AudioSegment.from_wav(path) 
    
    chunks = split_on_silence(song, 

        min_silence_len = 300, 
        silence_thresh = sil_thresh
    ) 
    try:
        os.system('rm -rf /home/mohiitaa/ufonia/phone_number_identification_from_speech/data/audio/' + n)
    except:
        pass
    try: 
        os.mkdir('/home/mohiitaa/ufonia/phone_number_identification_from_speech/data/audio/'+ n) 
    except(FileExistsError): 
        pass


    os.chdir('/home/mohiitaa/ufonia/phone_number_identification_from_speech/data/audio/' + n) 
    i = 0
    for chunk in chunks: 
            

        chunk_silent = AudioSegment.silent(duration = 10) 
        audio_chunk = chunk_silent + chunk + chunk_silent 
        audio_chunk.export("./chunk_{0}.wav".format(i), bitrate ='192k', format ="wav") 
        filename = 'chunk_'+str(i)+'.wav'
        i+=1
    os.chdir('..') 

def extract_mfcc(in_path, file, fmax, nMel, n):
    y, sr = librosa.load(in_path + file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nMel, fmax=fmax)
    
    plt.figure(figsize=(3, 3), dpi=100)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), fmax=fmax)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    try:
        os.system('rm -rf /home/mohiitaa/ufonia/phone_number_identification_from_speech/data/images/' + n)
    except:
        pass

    try: 
        os.mkdir('/home/mohiitaa/ufonia/phone_number_identification_from_speech/data/images/'+ n) 
    except(FileExistsError): 
        pass
    plt.savefig('/home/mohiitaa/ufonia/phone_number_identification_from_speech/data/images/' + n + '/' + file[:-3] + 'png', bbox_inches='tight', pad_inches=-0.1)
    
    plt.close()   
    return


def print_number(folder, model):

    phone_number = {}
    for file in os.listdir(folder):
        img = cv2.imread(folder + '/' + file)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        # img = img.to(device)
        outputs = model(img)
        _, predicted = torch.max(torch.unsqueeze(outputs.data,0), 1)
        phone_number[int(file.split("_")[1].split(".")[0])] =predicted.item()

    return phone_number

