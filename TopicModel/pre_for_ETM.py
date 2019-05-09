import os
import json
import sys
# import csv
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
from utils.data_provider import VQADataProvider



# q_a_i_path = os.path.join(root_path, "data/train/All_QA_Pairs_train.txt")
q_a_i_path = os.path.join(root_path, "data/train_val/All_QA_Pairs_train_val.txt")
q_voc_path = os.path.join(root_path, "vocab/question_vocab.json")
wordlist_path = os.path.join(root_path, "TopicModel/word.txt")
doc_path = os.path.join(root_path, "TopicModel/sentences.txt")

def gen_txt():
	with open(q_voc_path, "r") as f:
		q_dic = json.load(f)

	word_list = []

	exc_list = ["<break>", "<END>", "<START>", "<UNKNOWN>", "<UNK>"]

	for k, _ in q_dic.items():
		# exclude <break>, <END>, <START>, <UNKNOWN>
		if(k not in exc_list):
			word_list.append(k)


	sent_list = []
	_, raw_ques, _ = VQADataProvider.load_raw_iqa(q_a_i_path)
	for ques in raw_ques:
		sent_list.append(VQADataProvider.text_to_list(ques))
	# with open(q_a_i_path, "r") as csvfile:
	# 	# QA = csv.reader(csvfile, delimiter="\t", quotechar='\n')
	# 	for row in QA:
	# 		sent_list.append(data_provider.VQADataProvider.seq_to_list(row[2]))



	sent_idx_list = []
	for sent in sent_list:
		sent_idx_list.append([word_list.index(x) for x in sent if x not in exc_list])

	with open(wordlist_path, "w") as f:
		for item in word_list:
			f.write("%s\n"%item)

	with open(doc_path, "w") as f:
		for sent in sent_idx_list:
			f.write(" ".join([str(i) for i in sent]) + "\n")

if __name__ == "__main__":
	gen_txt()