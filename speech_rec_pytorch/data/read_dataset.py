import os
import argparse
import copy
import json

folder_name= ['train', 'test']

dir_numbers = list(range(0,10))


train_list = []
test_list = []

for fn in folder_name:

	total_path = fn + '/'

	for file in os.listdir(total_path):
		if fn == 'train':
			train_list.append(file)
		else:
			test_list.append(file)

partition = {}
partition['train'] = train_list
partition['test'] = test_list

labels = {}
for i in train_list:
	label_this = int(i.split("_")[0])
	labels[i] = label_this

for i in test_list:
	label_this = int(i.split("_")[0])
	labels[i] = label_this


with open('partition_aug.json', 'w') as fp:
    json.dump(partition, fp)

with open('labels_aug.json', 'w') as fp:
    json.dump(labels, fp)




