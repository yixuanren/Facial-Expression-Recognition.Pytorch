import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

#import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from pdb import set_trace


class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

net = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

cut_size = 44


batch = False

if batch:
	transform_test = transforms.Compose([
		transforms.Grayscale(3),
		transforms.Resize(48),
		transforms.TenCrop(cut_size),
		transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
	])
	
	dataroot = '/data/workplace/CelebA'
	test_set = datasets.ImageFolder(root=dataroot, transform=transform_test)
	
	dataloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

else:
	transform_test = transforms.Compose([
		transforms.TenCrop(cut_size),
		transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
	])
	
	def rgb2gray(rgb):
		return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
	
#	dataroot = '/data/workplace/CelebA/128_crop'
	dataroot = '/data/workplace/KDEF_crop562'
	test_set = sorted(os.listdir(dataroot))


# WARNING: Mini-batch and single-image inference can give different outputs!
# Haven't figured out why yet...

rep_list = []
#preds = np.zeros(len(test_set), dtype=np.uint8)


if batch:
	for i, data in enumerate(tqdm(dataloader)):
		inputs, _ = data
		
	#	set_trace()
		
		bs, ncrops, c, h, w = np.shape(inputs)
		
		inputs = inputs.view(-1, c, h, w)
		inputs = inputs.cuda()
		with torch.no_grad():
			rep, outputs = net(inputs)
		
	#	set_trace()
		
		outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
		rep_avg = rep.view(bs, ncrops, -1).mean(1)  # avg over crops
		
		score = F.softmax(outputs_avg, dim=1)
		_, predicted = torch.max(outputs_avg.data, 1)
		
		set_trace()
		
#		preds[i*64:i*64+bs] = predicted.cpu().numpy()  # For filtering to get a balanced subset
	
#		rep_list.append(score.cpu().tolist())  # Version 1: the output of the last layer (fc) before softmax
		rep_list.append(rep_avg.cpu().tolist())  # Version 2: the output of the second last layer (after activation)
	
else:
	for i in tqdm(xrange(len(test_set))):
		
		filename = test_set[i]
		raw_img = io.imread(dataroot+'/'+filename)
		gray = rgb2gray(raw_img)
		gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
		
		img = gray[:, :, np.newaxis]
		
		img = np.concatenate((img, img, img), axis=2)
		img = Image.fromarray(img)
		inputs = transform_test(img)
		
		ncrops, c, h, w = np.shape(inputs)
		
		inputs = inputs.view(-1, c, h, w)
		inputs = inputs.cuda()
		with torch.no_grad():
			rep, outputs = net(inputs)
		
	#	set_trace()
		
		outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
		rep_avg = rep.view(ncrops, -1).mean(0)  # avg over crops
		
		score = F.softmax(outputs_avg, dim=0)
		_, predicted = torch.max(outputs_avg.data, 0)
		
#		set_trace()
		
#		preds[i] = predicted.cpu().numpy()  # For filtering to get a balanced subset
	
#		rep_list.append(score.cpu().tolist())  # Version 1: the output of the last layer (fc) before softmax
		rep_list.append(rep_avg.cpu().tolist())  # Version 2: the output of the second last layer (after activation)

set_trace()
'''
np.savez_compressed('preds', preds)
print('preds.npz saved')
'''
np.savez_compressed('rep_list', rep_list)
print('rep_list.npz saved')
