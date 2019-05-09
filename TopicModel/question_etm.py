import numpy as np
import sys 
import os
import json

root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
# from utils.data_provider import VQADataProvider

# 10 topics
topic_path = os.path.join(root_path, "TopicModel/train_val_STTP_topics=10topWords=30times=5.txt")
q_a_i_path = os.path.join(root_path, "data/train/All_QA_Pairs_train_val.txt")
# save path
ques_topic_path = os.path.join(root_path, "TopicModel/question_topic.txt")

# def gen_ques_topic_distribution():
# 	# load all topics
# 	topics = []
# 	with open(topic_path, "r") as f:
# 		for line in f:
# 			if(line not in ['\n', '\r\n']):
# 				topics.append([x for x in line.split()])
# 	num_topic = len(topics)
# 	# load all questions
# 	_, all_question, _ = VQADataProvider.load_raw_iqa(q_a_i_path)
# 	num_ques = len(all_question)

# 	ques_topic_matrix = np.zeros((num_ques, num_topic))

# 	for i in range(num_ques):
# 		word_ls = VQADataProvider.text_to_list(all_question[i])
# 		for word in word_ls:
# 			for j in range(num_topic):
# 				if(word in topics[j]):
# 					ques_topic_matrix[i, j] += 1
# 	# normalize each row of the matrix
# 	row_sum = ques_topic_matrix.sum(axis=1)
# 	ques_topic_matrix = ques_topic_matrix/row_sum[:, np.newaxis]
# 	# save the question topic distribution
# 	np.savetxt(ques_topic_path, ques_topic_matrix, fmt="%2.3f")
# 	print("question topic distribution saved to %s"%ques_topic_path)


# input is a list of word that has been processed by VQADataProvidertext_to_list()
def etm_topic_distrib(word_ls):
	# load all topics
	topics = []
	with open(topic_path, "r") as f:
		for line in f:
			if(line not in ['\n', '\r\n']):
				topics.append([x for x in line.split()])
	num_topic = len(topics)

	q_topic = np.zeros(num_topic)
	for word in word_ls:
		for i in range(num_topic):
			if(word in topics[i]):
				q_topic[i] += 1
	q_topic = q_topic/q_topic.sum()
	return q_topic


if __name__ == "__main__":
	gen_ques_topic_distribution()
