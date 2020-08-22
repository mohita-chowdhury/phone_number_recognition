import torch
import os
import model
import cv2
import collections
PATH = '/scratch/shared/nfs1/mohita/ufonia/speech_rec_pytorch/saved/models/speech_net_aug.pth.tar'
device = torch.device("cuda")

def print_number(folder, model):

	phone_number = {}
	for file in os.listdir(folder):
		img = cv2.imread(folder + '/' + file)
		img = torch.from_numpy(img)
		img = torch.unsqueeze(img, 0)
		img = img.to(device)
		# import pdb; pdb.set_trace()
		outputs = model(img)
		_, predicted = torch.max(torch.unsqueeze(outputs.data,0), 1)
		phone_number[int(file.split("_")[1].split(".")[0])] =predicted.item()
		# print([file, predicted.item()])
	return phone_number


folder_name = '/users/mohita/nfs1_mohita/ufonia/phone_number_identification_from_speech/data/images/pn2'
speech_model = model.SpeechConv()
speech_model.to(device)
speech_model.load_state_dict(torch.load(PATH))
phone_number = print_number(folder_name, speech_model)


od = collections.OrderedDict(sorted(phone_number.items()))
output = []
for k, v in od.items():
	output.append(v)
print(output)
