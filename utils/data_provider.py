import random

import torch.utils.data as data
import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
import numpy as np
import pickle
import re
from PIL import Image
from gensim.models import KeyedVectors # load word2vec
import torch.utils.data as data
from torchvision import transforms
import config
from utils.mapping_category import load_mapping_ls
from TopicModel.question_etm import etm_topic_distrib


class VQADataProvider():
	def __init__(self):
		self.root_path = os.path.join(os.getenv("HOME"), "vqa2019")

	# load questions and set label for each one, preprocess for question_classifier
	@staticmethod
	def label_ques(folder):
		root_path = os.path.join(os.getenv("HOME"), "vqa2019")
		file1 = os.path.join(root_path, "data/%s/QAPairsByCategory/C1_Modality_%s.txt" % (folder, folder))
		file2 = os.path.join(root_path, "data/%s/QAPairsByCategory/C2_Plane_%s.txt" % (folder, folder))
		file3 = os.path.join(root_path, "data/%s/QAPairsByCategory/C3_Organ_%s.txt" % (folder, folder))
		file4 = os.path.join(root_path, "data/%s/QAPairsByCategory/C4_Abnormality_%s.txt" % (folder, folder))
		# for debug
		# print(file1)
		# print(file2)
		# print(file3)
		# print(file4)
		file_ls = [file1, file2, file3, file4]
		questions = []
		ques_categories = []

		counter = 0
		for file in file_ls:
			counter += 1
			ques_ls, num_ques = VQADataProvider.load_question(file)
			questions += ques_ls
			ques_categories += [counter]*num_ques

		# shuffle
		random.seed(3)
		pair = list(zip(questions, ques_categories))
		random.shuffle(pair)
		questions, ques_categories = zip(*pair)

		return questions, ques_categories

	@staticmethod
	def load_question(filename):
		ques_ls = []
		with open(filename, "r") as f:
			# num_row = 0
			for row in f:
				qa = row.rstrip().split("|")
				ques_ls.append(qa[1])
				# num_row += 1
		# use set instead of list
		ques_set = list(set(ques_ls))
		# return ques_ls, num_row
		return ques_set, len(ques_set)

	# load the file
	@staticmethod
	def load_raw_iqa(filename):
		ques_ls = []
		ans_ls = []
		imgid_ls = []
		with open(filename, "r") as f:
			for row in f:
				iqa = row.rstrip().split("|")
				imgid_ls.append(iqa[0])
				ques_ls.append(iqa[1])
				ans_ls.append(iqa[2])
		return imgid_ls, ques_ls, ans_ls


	# lower and remove punctuations
	@staticmethod
	def text_to_list(s):
		t_str = s.lower()
		# for i in [r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\/']:
		# 	t_str = re.sub(i, ' ', t_str)
		# for i in [r'\.', r'\;', r'\?']:
		# 	t_str = re.sub(i, ' <BREAK> ', t_str)
		for i in [r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\/', r'\.', r'\;', r'\?']:
			t_str = re.sub(i, ' ', t_str)
		for i in [r'\-']:
			t_str = re.sub(i, '', t_str)
		w_list = re.sub(r'\?', '', t_str.lower()).split(' ')
		w_list = list(filter(lambda x: len(x) > 0, w_list))
		# w_list = ['<START>'] + w_list + ['<END>']
		w_list = w_list + ['<END>']
		return w_list

	@staticmethod
	def qlist_to_matrix(raw_q_list, max_length, EMB, embedding_size=200):
		''' 
		convert question string to embedding matrix
		'''
		q_list = raw_q_list[:-1]
		MED_matrix = np.zeros((max_length, embedding_size))
		for i in range(max_length):
			if i >=len(q_list):
				break
			else:
				w = q_list[i]
				if w not in EMB.wv.vocab:
					MED_matrix[i] = np.zeros(embedding_size)
				else:
					MED_matrix[i] = EMB[w]
		return MED_matrix

	@staticmethod
	def qlist_to_indices(raw_q_list, max_length, ques_word_mapping, mode):
		''' 
		convert question string to list of indices
		-1 means the word is <UNKNOWN> for training process
		'''
		q_list = raw_q_list[:-1]
		# idx_ls = [0]*max_length
		idx_ls = np.zeros(max_length)
		for i in range(max_length):
			if(i>=len(q_list)):
				break
			w = q_list[i]
			# if(mode != "test"):
			if w not in ques_word_mapping:
				idx_ls[i] = -1
			else:
				j = ques_word_mapping.index(w)
				idx_ls[i] = j
			# else:
			# 	if w not in ques_word_mapping:
			# 		w = "<UNK>"
			# 	j = ques_word_mapping.index(w)
			# 	idx_ls[i] = j

		return idx_ls
		

	@staticmethod
	def alist_to_vec(a_list, dimension, max_len, ans_vocab):
		# ans_vocab is a list of word
		# use the index of word as value of vector
		a_vec = np.zeros((max_len, dimension))
		for i in range(max_len):
			if(i>=len(a_list)):
				break
			else:
				w = a_list[i]
				if(w not in ans_vocab):
				    w = "<UNK>"
				# j = ans_vocab[w]
				j = ans_vocab.index(w)
				a_vec[i, j] = 1
		return a_vec


	@staticmethod
	def adjust_img(img):
		w, h = img.size
		if(w<h):
			one_img = img.crop((0, 0, w, int(h/3*2)))
			two_img = img.crop((0, int(h/3*2), w, h))
			new_img = Image.new("RGB", (w*2, int(h/3*2)))
			new_img.paste(one_img)
			new_img.paste(two_img, (w, int(h/6)))
			return new_img
		else:
			one_img = img.crop((0, 0, int(w/3*2), h))
			two_img = img.crop((int(w/3*2), 0, w, h))
			new_img = Image.new("RGB", (int(w/3*2), h*2))
			new_img.paste(one_img)
			new_img.paste(two_img, (int(w/6), h))
			return new_img

	@staticmethod
	def loadMED_Embedding(embedding_path):
		print("Loading MED Embedding ......")
		MED = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
		print("Done." + str(len(MED.wv.vocab)) + " words loaded!")
		return MED

	@staticmethod
	def load_test_data(test_path):
		test_img_ids = []
		test_questions = []
		with open(test_path, "r") as f:
			for row in f:
				iq = row.rstrip().split("|")
				test_img_ids.append(iq[0])
				test_questions.append(iq[1])
		assert len(test_img_ids) == len(test_questions), "images number and questions should be same!"
		assert len(test_img_ids) == 500, "Number of records should be 500!"
		return test_img_ids, test_questions


class VQADataset(data.Dataset):
	# def __init__(self, mode, opt, EMB=None):
	def __init__(self, mode, opt):
		self.opt = opt
		# if(EMB is None):
		# 	self.EMB = VQADataProvider.loadMED_Embedding(self.opt.EMBEDDING_PATH)
		# else:
		# 	self.EMB = EMB
		# self.EMBEDDING_SIZE = 200
		self.mode = mode

		# load question word mapping
		ques_word_map_path = os.path.join(root_path, "embedding/embed_mapping.pkl")
		with open(ques_word_map_path, "rb") as q_w_f:
			self.ques_word_ls = pickle.load(q_w_f)

		if(self.mode != "test"):
			# load imgid, question, answer
			self.raw_data_path = os.path.join(root_path, "data/%s/All_QA_Pairs_%s.txt" % (self.mode, self.mode))
			self.img_ids, self.questions, self.answers = VQADataProvider.load_raw_iqa(self.raw_data_path)
		else:
			self.raw_data_path = os.path.join(root_path, "data/test/VQAMed2019_Test_Questions.txt")
			self.img_ids, self.questions = VQADataProvider.load_test_data(self.raw_data_path)

		if(self.mode != "test"):
			# question category (4 category)
			all_ques_set, ques_categories = VQADataProvider.label_ques(self.mode)
			self.ques_categs = []
			for ques in self.questions:
				idx = all_ques_set.index(ques)
				self.ques_categs.append(ques_categories[idx])

		# # load question topic distribution
		# ques_etm_path = os.path.join(root_path, "TopicModel/question_topic.txt")
		# self.ques_etm_topics = np.loadtxt(ques_etm_path, dtype=float)
		# load the function of creating etm topic distribution
		self.gen_q_topic_distribution = etm_topic_distrib

		# load answer sub-category (70 including abnormality)
		self.ans_mapping_classes = load_mapping_ls()
		if(mode != "test"):
			# categorize answers (index in the mapping classes) (if not in, set the class to the last one)
			self.ans_categories = []
			for ans in self.answers:
				if(ans not in self.ans_mapping_classes):
					self.ans_categories.append(self.opt.ALL_CLASS_NUM - 1)
				else:
					self.ans_categories.append(self.ans_mapping_classes.index(ans))

		# load vocabulary for abnormality cateogy
		vocab_path = os.path.join(root_path, "vocab/answer_vocab.pkl")
		with open(vocab_path, "rb") as f:
			self.answer_vocab = pickle.load(f)

		# img transforms
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.029, 0.224, 0.225])
		if(mode == "train"):
			self.transformations = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), self.normalize, ])
		else:
			self.transformations = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), self.normalize])


		# img folder 
		if(self.mode != "test"):
			self.img_folder = os.path.join(root_path, "data/%s/%s_images"%(self.mode, self.mode))
		else:
			self.img_folder = os.path.join(root_path, "data/test/VQAMed2019_Test_Images")

		self.data_len = len(self.questions)


	def __getitem__(self, index):
		# image 
		img_id = self.img_ids[index]
		image_path = os.path.join(self.img_folder, "%s.jpg"%img_id)
		image = Image.open(image_path).convert("RGB")
		# adjust long image to approximately square image
		ratio = image.size[0]/image.size[1]
		if(ratio>self.opt.IMG_RATIO_THRESHOLD or ratio<(1/self.opt.IMG_RATIO_THRESHOLD)):
			image = VQADataProvider.adjust_img(image)
		img_matrix = self.transformations(image)

		# question embedding
		q_list = VQADataProvider.text_to_list(self.questions[index])
		q_idx_ls = VQADataProvider.qlist_to_indices(q_list, self.opt.MAX_WORDS_IN_QUESTION, self.ques_word_ls, self.mode)
		if(self.mode != "test"):
			# question category
			q_category_gt = np.zeros(self.opt.QUES_CLASS_NUM)
			q_raw_category = self.ques_categs[index]
			q_category_gt[q_raw_category - 1] = 1
		# question etm topic
		# q_etm_topic = ques_etm_topics[index, :]
		q_etm_topic = self.gen_q_topic_distribution(q_list)

		if(self.mode != "test"):
			# answer classification
			# input of pytorch CrossEntropyLoss is integer, not a 0/1 vector
			# but this answer classification vector is also used for loss conputing
			ans_category_gt = np.zeros(self.opt.ALL_CLASS_NUM)
			ans_raw_category = self.ans_categories[index]
			ans_category_gt[ans_raw_category] = 1
			# answer producing
			# if question is not abnormality category
			if(q_raw_category != 4):
				ans_vec_gt = np.zeros((self.opt.MAX_WORDS_IN_ANSWER, self.opt.NUM_OUTPUT_UNITS))
			else:
				ans_list = VQADataProvider.text_to_list(self.answers[index])
				ans_vec_gt = VQADataProvider.alist_to_vec(ans_list, self.opt.NUM_OUTPUT_UNITS, self.opt.MAX_WORDS_IN_ANSWER, self.answer_vocab)

		# self.questions[index] is used to predict question category
		# during training and validation, use the q_cateogry_gt as the input because the SVM is 100% accurate
		# during test, use self.questions[index] to predict q_category which is used as the input of the model
		if(self.mode != "test"):
			return img_matrix, q_idx_ls, q_raw_category, q_category_gt, q_etm_topic, ans_raw_category, ans_category_gt, ans_vec_gt#, self.questions[index]
		else:
			return img_id, img_matrix, q_idx_ls, q_etm_topic, self.questions[index]

	def __len__(self):
		return self.data_len








if __name__ == "__main__":
	questions, classes = VQADataProvider.label_ques("train")
	print("# of questions:, ", len(questions))
	print("# of classes: ", len(classes))
	print(questions[100:105])
	print(classes[100:105])