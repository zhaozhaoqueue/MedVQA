import json
import random
import re
from PIL import Image
import numpy as np
import pandas as pd
import torch.utils.data as data
from torchvision import transforms
import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
from utils.data_provider import VQADataProvider


class TuneDataset(data.Dataset):
	def __init__(self, opt, mode):
		self.opt = opt
		self.mode = mode
		self.img_folder = os.path.join(root_path, "data/%s/%s_images"%(mode, mode))
		# img preprocessing
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.029, 0.224, 0.225])
		self.transformations = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), self.normalize, ])

		self.label_path = os.path.join(root_path, "tune_imgnet/img_topic.json")
		with open(self.label_path, "r") as f:
			img_lab_dic = json.load(f)
		# self.img_labvec_list = []
		self.img_lab_list = []
		for img, lab in img_lab_dic.items():
			# self.img_labvec_list.append(("%s.jpg"%img, [float(x) for x in lab_vec]))
			# self.img_lab_list.append(("%s.jpg"%img, lab_vec.index(max(lab_vec))))
			self.img_lab_list .append(("%s.jpg"%img, int(lab)))


	def __getitem__(self, index):
		# (img_path, label_vec) = self.img_lab_list[index]
		(img_name, label) = self.img_lab_list[index]
		# image_path = os.path.join(root_path, self.img_folder, img_path)
		image_path = os.path.join(self.img_folder, img_name)
		image = Image.open(image_path).convert('RGB')

		# adjust long image to approximately square image
		ratio = image.size[0]/image.size[1]
		if(ratio>self.opt.IMG_RATIO_THRESHOLD or ratio<(1/self.opt.IMG_RATIO_THRESHOLD)):
			image = VQADataProvider.adjust_img(image)

		img_vec = self.transformations(image)
		# return img_vec, np.asarray(label_vec)
		return img_vec, label
	def __len__(self):
		return len(self.img_lab_list)