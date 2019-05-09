import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(os.path.join(root_path, "utils"))
sys.path.append(os.path.join(root_path, "models"))
import time

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision

# from models.coatt_model import mfh_coatt_Med
from models.coatt_model_emb import mfh_coatt_Med_emb
from models.question_classifier import predict_q_category
from utils.data_provider import VQADataProvider, VQADataset
import config

def make_prediction(model, opt):
	test_data = VQADataset(mode="test", opt=opt)
	test_loader = data.DataLoader(dataset=test_data, shuffle=False, batch_size=500, num_workers=1)
	model.cuda()
	model.eval()
	# set random seed for sampling answers
	# np.random.seed(3)

	img_id_ls = []
	ques_category = []
	ans_classification = []
	ans_predict = []
	for step, (img_id, img_matrix, q_idx_ls, q_etm_topic, q_raw) in enumerate(test_loader):
		# for debug
		# print(img_id)
		# img_id_ls += img_id.tolist()
		img_id_ls += list(img_id)
		# predict question category (4 categories)
		# q_raw_category = predict_q_category(q_raw.tolist())
		q_raw_category = predict_q_category(list(q_raw))
		ques_category += q_raw_category.tolist()
		# q_num = q_raw_category.size()[0]
		q_num = len(q_raw_category)
		# one-hot encoding question category
		q_category = np.zeros((q_num, opt.QUES_CLASS_NUM))
		for i in range(q_num):
			qc = q_raw_category[i] - 1
			q_category[i, qc] = 1
		q_category = torch.from_numpy(q_category).cuda().float()

		img_matrix = img_matrix.cuda().float()
		q_idx_ls = q_idx_ls.cuda().long()
		q_etm_topic = q_etm_topic.cuda().float()


		classification, ans_producing = model(img_matrix, q_idx_ls, q_category, q_etm_topic, mode="test")

		classification = classification.cpu().detach()
		classification = F.softmax(classification, dim=1)
		class_pred = torch.argmax(classification, dim=1)
		ans_classification += class_pred.tolist()

		ans_producing = ans_producing.cpu().detach().numpy()
		ans_producing = np.exp(ans_producing)
		for i in range(ans_producing.shape[0]):
			seed = 3
			words = []
			while(len(words) == 0):
				for j in range(ans_producing.shape[1]):
					a_idx = np.random.choice(opt.NUM_OUTPUT_UNITS, size=1, replace=False, p=ans_producing[i, j, :])[0]
					words.append(test_data.answer_vocab[a_idx])
				if("<END>" in words):
					words = words[:words.index("<END>")]
				# if("<START>" in words):
				# 	words.remove("<START>")
				if("<UNK>" in words):
					words.remove("<UNK>")
				# if("<break>" in words):
				# 	words.remove("<break>")
			ans_predict.append(" ".join(words))

	# update the answer which is not predicted to be abnormality category
	# for i in range(len(ans_classification)):
	# 	if(ans_classification[i] != opt.ALL_CLASS_NUM - 1):
	# 		ans_predict[i] = test_data.ans_mapping_classes[ans_classification[i]]
	print("question classification")
	print(ques_category)
	print("answer classification")
	print(ans_classification)
	for i in range(len(ques_category)):
		# print(1)
		if(ques_category[i] != opt.QUES_CLASS_NUM and ans_classification[i] != opt.ALL_CLASS_NUM - 1):
			ans_predict[i] = test_data.ans_mapping_classes[ans_classification[i]]

	return img_id_ls, ans_predict


def save_prediction(img_id_ls, ans_predict):
	submission_path = os.path.join(root_path, "results/submission/%s-submission.txt" % time.strftime("%m%d%H"))
	with open(submission_path, "w") as f:
		for (img_id, ans) in zip(img_id_ls, ans_predict):
			f.write("%s|%s\n" % (img_id, ans))
	print("submission file is saved at %s" % submission_path)

if __name__ == "__main__":
	opt = config.parse_opt()
	# load the question word embedding matrix
	embedding_matrix = np.load(os.path.join(root_path, "embedding/embedding_matrix.npy"))
	embedding_matrix_tensor = torch.from_numpy(embedding_matrix).float()

	model = mfh_coatt_Med_emb(opt, embedding_matrix_tensor)
	# load model
	model_path = os.path.join(root_path, "models/checkpoints/2019050811-100-iteration-model.pt")
	# model_path = os.path.join(root_path, "models/checkpoints/2019050711-200-iteration-model.pt")
	model.load_state_dict(torch.load(model_path))
	model.cuda()
	model.eval()
	img_id_ls, ans_predict = make_prediction(model, opt)

	# save file
	save_prediction(img_id_ls, ans_predict)
