import torch
import model
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device("cuda")

def test(test_gen, model, gpu_available):
	correct = 0
	total = 0
	pred = []
	true = []
	with torch.no_grad():
	    for data in test_gen:
	        images, labels = data
	        if gpu_available == 't':
		        images, labels = images.to(device), labels.to(device)
	        outputs = model(images)
	        # import pdb; pdb.set_trace()
	        _, predicted = torch.max(torch.unsqueeze(outputs.data,0), 1)
	        # print([labels, predicted])
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()
	        pred.append(predicted.item())
	        true.append(labels.item())


	print('The accuracy of the network on the 100 test images: %d %%' % (
	    100 * correct / total))

	print('-----------CONFUSION MATRIX-------------')
	conf_matrix = confusion_matrix(true, pred)
	print(conf_matrix)
	hm = sns.heatmap(conf_matrix, annot=True)
	figure = hm.get_figure()
	
	figure.savefig("./saved/metrics/confusion_matrix.png", dpi=400)
	print('-----------MODEL----------------')
	print(model)