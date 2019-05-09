import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score

import string
import nltk
import warnings
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# from nltk.corpus import wordnet as wn

import torch
from torch import from_numpy
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data

from utils.data_provider import VQADataProvider, VQADataset
import config
# from models.coatt_model import mfh_coatt_Med
from models.coatt_model_emb import mfh_coatt_Med_emb



def evaluate(model, opt, remove_stopwords=False, save_path=None):
	model.cuda()
	model.eval()

	val_data = VQADataset(mode="val", opt=opt)
	val_loader = data.DataLoader(dataset=val_data, shuffle=False, batch_size=opt.VAL_BATCH_SIZE, num_workers=1)

	# set seed
	# np.random.seed(3)
	
	ans_classification = []
	ans_predict = []
	ques_category = []
	for step, (img_matrix, q_idx_ls, q_raw_category, q_category_gt, q_etm_topic, ans_raw_category, ans_category_gt, _) in enumerate(val_loader):
		# print("validation step %d start" % step)
		# step_classification = []
		# step_producing = []
		ques_category += list(q_raw_category)
		# move to GPU
		img_matrix = img_matrix.cuda().float()
		q_idx_ls = q_idx_ls.cuda().long()
		q_category_gt = q_category_gt.cuda().float()
		q_etm_topic = q_etm_topic.cuda().float()
		ans_raw_category = ans_raw_category.cuda().long()
		# ans_vec_gt = ans_vec_gt.cuda().float()

		classification, ans_producing = model(img_matrix, q_idx_ls, q_category_gt, q_etm_topic, mode="val")

		classification = classification.cpu().detach()
		classification = F.softmax(classification, dim=1)
		class_pred = torch.argmax(classification, dim=1)
		ans_classification += class_pred.tolist()

		ans_producing = ans_producing.cpu().detach().numpy()
		# ans_producing = torch.exp(ans_producing)
		ans_producing = np.exp(ans_producing)

		# for i in range(ans_producing.size()[0]):
		for i in range(ans_producing.shape[0]):
			seed = 3
			words = []
			# to avoid the situation that answer only has <END> or <UNK>
			while(len(words) == 0):
				seed += 1
				np.random.seed(seed)
				# for j in range(ans_producing.size()[1]):
				for j in range(ans_producing.shape[1]):
					a_idx = np.random.choice(opt.NUM_OUTPUT_UNITS, size=1, replace=False, p=ans_producing[i, j, :])[0]
					words.append(val_data.answer_vocab[a_idx])
				if("<END>" in words):
					words = words[:words.index("<END>")]
				# if("<START>" in words):
				# 	words.remove("<START>")
				if("<UNK>" in words):
					words.remove("<UNK>")
				# if("<break>" in words):
				# 	words.remove("<break>")
			ans_predict.append(" ".join(words))

	# update the answer which is not abnormality category (mapping back)
	# for i in range(len(ans_classification)):
	# 	if(ans_classification[i] != opt.ALL_CLASS_NUM - 1):
	# 		ans_predict[i] = val_data.ans_mapping_classes[ans_classification[i]]
	for i in range(len(ans_predict)):
		if(ques_category[i] != opt.QUES_CLASS_NUM and ans_classification[i] != (opt.ALL_CLASS_NUM - 1)):
			ans_predict[i] = val_data.ans_mapping_classes[ans_classification[i]]

	# compute confusion matrix for classification -- done
	ans_categories = val_data.ans_categories
	c_m, acc = compute_classify_result(ans_categories, ans_classification)

	# process raw answers
	gt_answers = val_data.answers
	# for raw_ans in val_data.answers:
		# processed_ans = VQADataProvider.text_to_list(raw_ans)
		# processed_ans = " ".join(processed_ans)
		# gt_answers.append(processed_ans)

	# compute bleu for the answers in abnormality category
	abnor_idx_ls = []
	for i in range(len(ans_categories)):
		if(ans_categories[i] == (opt.ALL_CLASS_NUM - 1)):
			abnor_idx_ls.append(i)
	gt_answers_abnor = [gt_answers[k] for k in abnor_idx_ls]
	pred_answers_abnor = [ans_predict[k] for k in abnor_idx_ls]
	abnor_bleu = compute_bleu(gt_answers_abnor, pred_answers_abnor, remove_stopwords=False)

	# # maybe compute bleu for all answers
	# all_bleu = compute_bleu(gt_answers, ans_predict)
	# all_bleu_with_sw = compute_bleu(gt_answers, ans_predict, remove_stopwords=remove_stopwords)

	# save the answers and classification to the files -- done
	if save_path != None:
		save_file(ans_classification, ans_predict, save_path)

	return ans_classification, c_m, acc, ans_predict, abnor_bleu#, all_bleu


def compute_classify_result(gt, prediction):
	# comfusion matrix
	c_m = confusion_matrix(gt, prediction)
	# accuracy
	acc = accuracy_score(gt, prediction)
	return c_m, acc


'''
Arguments:
	gt: list of ground truth answers (processed by text_to_list() and consolidated to a sentence)
	prediction: list of predicted answers
	case_sensitive: boolean
	remove_stopwords: boolean, if remove the stopwords or not
	stem: boolean, if stem the words or not
'''
def compute_bleu(gt, prediction, case_sensitive=False, remove_stopwords=False, stem=False):
	# Hide warnings
	warnings.filterwarnings('ignore')

	# English Stopwords
	stops = set(stopwords.words("english"))
	# Stemming
	stemmer = SnowballStemmer("english")
	# Remove punctuation from string
	translator = str.maketrans('', '', string.punctuation)

	# define max score and current score
	max_score = len(gt)
	current_score = 0

	assert len(gt) == len(prediction), "Ground truth and prediction should be same length! %d != %d" % (len(gt), len(prediction))

	# for debug
	pair = []

	for gt_ans, pred_ans in zip(gt, prediction):
		if not case_sensitive:
			gt_ans = gt_ans.lower()
			pred_ans = pred_ans.lower()

		# Split caption into individual words (remove punctuation)
		gt_ans = nltk.tokenize.word_tokenize(gt_ans.translate(translator))
		pred_ans = nltk.tokenize.word_tokenize(pred_ans.translate(translator))

		if remove_stopwords:
			gt_ans = [word for word in gt_ans if word.lower() not in stops]
			pred_ans = [word for word in pred_ans if word.lower() not in stops]

		if stem:
			gt_ans = [stemmer.stem(word) for word in gt_ans]
			pred_ans = [stemmer.stem(word) for word in pred_ans]

		# calculate BLEU for current answer
		try:
			# If both the GT and candidate are empty, assign a score of 1 for this caption
			if(len(gt_ans) == 0 and len(pred_ans) == 0):
				bleu_score = 1
			else:
				# use 1-gram
				bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_ans], pred_ans, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method0)
		except ZeroDivisionError:
			raise Exception('Problem with {} {}', gt_ans, pred_ans)
			# pass
		current_score += bleu_score
		pair.append((gt_ans, pred_ans, bleu_score))

	# # for debug
	# print(pair)

	return current_score/max_score


def save_file(obj1, obj2, filepath):
	with open(filepath, "w") as f:
		for (ele1, ele2) in zip(obj1, obj2):
			f.write("%d|%s" % (ele1, ele2))
			f.write("\n")
	print("saved to %s" % filepath)


if __name__ == "__main__":
	opt = config.parse_opt()

	embedding_matrix = np.load(os.path.join(root_path, "embedding/embedding_matrix.npy"))
	embedding_matrix_tensor = torch.from_numpy(embedding_matrix).float()
	model = mfh_coatt_Med_emb(opt, embedding_matrix_tensor)
	# load model
	model_path = os.path.join(root_path, "models/pretrained/2019043015_model.pt")
	model.load_state_dict(torch.load(model_path))
	model.eval()

	ans_classification, c_m, acc, ans_predict, abnor_bleu = evaluate(model=model, opt=opt, remove_stopwords=False, save_path=None)
	# print(c_m)
	print(acc)
	print("abnor bleu and all bleu: ", abnor_bleu)
