from flask import Flask, render_template, request
import os
import utils as ml_model
import torch
import collections
import model
app = Flask(__name__)

AUDIO_DATA_PATH = "/home/mohiitaa/ufonia/phone_number_identification_from_speech/data/audio/test.wav"
PATH = '/home/mohiitaa/ufonia/phone_number_identification_from_speech/pretrained_model/speech_net_aug.pth.tar'

@app.route('/')
def home():
    return render_template('./home.html')

@app.route('/upload', methods=['POST'])
def upload_func():

    request.files['data'].save(AUDIO_DATA_PATH)
    print(request.files['data'])
    sil_thresh = 25
    ml_model.silence_based_conversion(AUDIO_DATA_PATH, 'test', -1*sil_thresh)
    # print('Audio split successful...')


    chunks_path = '/home/mohiitaa/ufonia/phone_number_identification_from_speech/data/audio/' + 'test' + '/'

    # import pdb; pdb.set_trace()
    print(os.listdir(chunks_path))
    for wavfile in os.listdir(chunks_path):
        S = ml_model.extract_mfcc(chunks_path, wavfile, 8000, 256, 'test')

    print('MFCC extracted...')
    spec_path = '/home/mohiitaa/ufonia/phone_number_identification_from_speech/data/images/' + 'test'
    speech_model = model.SpeechConv()
    # speech_model.to(device)
    speech_model.load_state_dict(torch.load(PATH))
    print_number = ml_model.print_number(spec_path, speech_model)
    print('Pretrained speech model loaded...')
    print('Loading predictions...')
    od = collections.OrderedDict(sorted(print_number.items()))
    output = []
    output2 = ''
    for k, v in od.items():
        output.append(v)
        output2 += str(v)
    print(output2)
    if output2 != '':
        return output2
    else: 
        return "Error"