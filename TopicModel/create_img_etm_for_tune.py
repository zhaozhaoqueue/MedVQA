import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
import numpy as np
import json
import config
from utils.data_provider import VQADataProvider
from TopicModel.question_etm import etm_topic_distrib

def label_img_with_ques_etm():
	opt = config.parse_opt()
	q_i_a_path = os.path.join(root_path, "data/train/All_QA_Pairs_train.txt")

	img_ques_dict = {}
	with open(q_i_a_path, "r") as f:
		for row in f:
			q_i_a = row.strip().split("|")
			img = q_i_a[0]
			ques = q_i_a[1]
			if(img in img_ques_dict):
				img_ques_dict[img].append(ques)
			else:
				img_ques_dict[img] = [ques]

	img_topic_dict = {}
	for img, qs in img_ques_dict.items():
		img_topic_vector = np.zeros(opt.ETM_TOP_NUM)
		for q in qs:
			words = VQADataProvider.text_to_list(q)
			q_t_v = etm_topic_distrib(words)
			img_topic_vector = np.add(img_topic_vector, q_t_v)
		img_topic_dict[img] = (np.argmax(img_topic_vector)).item()
	return img_topic_dict

if __name__ == "__main__":
	save_path = os.path.join(root_path, "tune_imgnet/img_topic.json")
	img_topic_dict = label_img_with_ques_etm()
	with open(save_path, "w") as f:
		json.dump(img_topic_dict, f)