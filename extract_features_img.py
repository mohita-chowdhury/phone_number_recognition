'''
Created on 2017.12.9

@author: Richard
'''

import os
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt

# number = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']



def extract_mfcc(in_path, file, fmax, nMel, n):
    y, sr = librosa.load(in_path + file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nMel, fmax=fmax)
    
    plt.figure(figsize=(3, 3), dpi=100)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), fmax=fmax)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # try:
    #     os.makedirs('number_images/'+ n)
    # except OSError as e:
    #     if e.errno != errno.EEXIST:
    #         raise
    plt.savefig('/users/mohita/nfs1_mohita/ufonia/phone_number_identification_from_speech/data/images/' + n + '/' + file[:-3] + 'png', bbox_inches='tight', pad_inches=-0.1)
    
    plt.close()   
    return


count = 0       # number of files processed

      # input directory
# in_path = '222/'
# numbers = ['zero','one','two','three','four','five','six','seven','eight','nine']
numbers = ['pn1', 'pn2']


for n in range(len(numbers)):
    in_path = '/users/mohita/nfs1_mohita/ufonia/phone_number_identification_from_speech/data/audio/' + numbers[n] + '/'
    for wavfile in os.listdir(in_path):
        
        # Input file
        S = extract_mfcc(in_path, wavfile, 8000, 256, numbers[n])
        
        # Count processed files
        count += 1
        if count % 20 == 0:
            print ('%d files processed.' % count)

print ('Done!\t%d files processed.' % count)
