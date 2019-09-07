"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

from torchvision import datasets
from tqdm import tqdm

from pdb import set_trace


class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

net = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

rep_list = []


cut_size = 44

transform_test = transforms.Compose([
	transforms.TenCrop(cut_size),
	transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#dataroot = '/data/workplace/CelebA_128crop_FD/CelebA'
#test_set = datasets.ImageFolder(root=dataroot, transform=transform_test)
#TODO: Mini-Batch Inference

dataroot = '/data/workplace/CelebA/128_crop'

for filename in tqdm(sorted(os.listdir(dataroot))):
	
	raw_img = io.imread(dataroot+'/'+filename)
	gray = rgb2gray(raw_img)
	gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
	
	img = gray[:, :, np.newaxis]
	
	img = np.concatenate((img, img, img), axis=2)
	img = Image.fromarray(img)
	inputs = transform_test(img)
	
#	set_trace()
	
	ncrops, c, h, w = np.shape(inputs)
	
	inputs = inputs.view(-1, c, h, w)
	inputs = inputs.cuda()
	with torch.no_grad():
		rep, outputs = net(inputs)
	
	outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
	rep_avg = rep.view(ncrops, -1).mean(0)  # avg over crops
	
	score = F.softmax(outputs_avg, dim=0)
	_, predicted = torch.max(outputs_avg.data, 0)
	
#	set_trace()
	
#	rep_list.append(score.cpu().tolist())  # Version 1: the output of the last layer (fc) before softmax
	rep_list.append(rep_avg.cpu().tolist())  # Version 2: the output of the second last layer (after activation)
	
	'''
	plt.rcParams['figure.figsize'] = (13.5,5.5)
	axes=plt.subplot(1, 3, 1)
	plt.imshow(raw_img)
	plt.xlabel('Input Image', fontsize=16)
	axes.set_xticks([])
	axes.set_yticks([])
	plt.tight_layout()
	
	plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
	
	plt.subplot(1, 3, 2)
	ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
	width = 0.4       # the width of the bars: can also be len(x) sequence
	color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
	for i in range(len(class_names)):
		plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
	plt.title("Classification results ",fontsize=20)
	plt.xlabel(" Expression Category ",fontsize=16)
	plt.ylabel(" Classification Score ",fontsize=16)
	plt.xticks(ind, class_names, rotation=45, fontsize=14)
	
	axes=plt.subplot(1, 3, 3)
	emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
	plt.imshow(emojis_img)
	plt.xlabel('Emoji Expression', fontsize=16)
	axes.set_xticks([])
	axes.set_yticks([])
	plt.tight_layout()
	# show emojis
	
	#plt.show()
	plt.savefig(os.path.join('images/results/l.png'))
	plt.close()
	
	print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
	'''

#set_trace()

np.savez_compressed('rep_list', rep_list)
print('rep_list.npz saved')
