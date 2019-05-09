import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
from utils.data_provider import VQADataProvider

def load_ques2set(filename):
	ques_set = set()
	with open(filename, "r") as f:
		for row in f:
			qa = row.rstrip().split("|")
			ques_set.add(qa[1])
	return ques_set

def load_all2set(filename):
	ques_set 	= set()
	ans_set 	= set()
	img_set 	= set()
	with open(filename, "r") as f:
		for row in f:
			qa = row.rstrip().split("|")
			if(len(qa) == 3):
				img_set.add(qa[0])
				ques_set.add(qa[1])
				ans_set.add(qa[2])
			else:
				print("Abnormal Row")
	return img_set, ques_set, ans_set


def check_raw_data(filename):
	answer_set = set()
	img_id_set = set()
	ques_set = set()
	q_i_pair_set = set()
	q_a_pair_set = set()
	with open(filename, "r") as f:
		# print("total q-i pair: ", len(f))
		counter = 0
		for row in f:
			qa = row.split("|")
			img_id_set.add(qa[0])
			ques_set.add(qa[1])
			answer_set.add(qa[2].rstrip())
			counter += 1
			q_i_pair_set.add((qa[0], qa[1]))
			q_a_pair_set.add((qa[1], qa[2].rstrip()))
		print("total records: ", counter)
	return answer_set, img_id_set, ques_set, q_i_pair_set, q_a_pair_set


def check_raw_readme(filename, ans_list):
	print("readme length: ", len(ans_list))
	raw_ans, _, _, _, _ = check_raw_data(filename)
	in_data_not_readme = raw_ans.difference(set(ans_list))
	in_readme_not_data = set(ans_list).difference(raw_ans)
	return in_data_not_readme, in_readme_not_data
	
def check_q_i_a_pair(filename):
	_, _, _, q_i_pairs, _ = check_raw_data(filename)
	q_i_dic_for_ans = {}
	with open(filename, "r") as f:
		for row in f:
			qa = row.rstrip().split("|")
			if((qa[0], qa[1]) in q_i_dic_for_ans):
				q_i_dic_for_ans[(qa[0], qa[1])].append(qa[2])
			else:
				q_i_dic_for_ans[(qa[0], qa[1])] = [qa[2]]
	return q_i_dic_for_ans

# this function does not work for all the answers
def check_multi_answer(filename):
	# counter = 0
	sep_answer_list = []
	sep_answer_set = []
	with open(filename, "r") as f:
		for row in f:
			qa = row.rstrip().split("|")
			ans = qa[2]
			ans_ls = [a.strip() for a in ans.split(",")]
			ans_set = set([a.strip() for a in ans.split(",")])
			sep_answer_list.append(ans_ls)
			if(ans_set not in sep_answer_set):
				sep_answer_set.append(ans_set)
	return sep_answer_list, sep_answer_set


# check if there is any overlapping question in different categories
def check_ques_overlap(file1, file2):
	ques_1_set = set()
	with open(file1, "r") as f1:
		for row in f1:
			qa = row.rstrip().split("|")
			ques_1_set.add(qa[1])

	ques_2_set = set()
	with open(file2, "r") as f2:
		for row in f2:
			qa = row.rstrip().split("|")
			ques_2_set.add(qa[1])
	return ques_1_set.intersection(ques_2_set)

def check_ques_completeness():
	path = os.path.join(root_path, "data/train/All_QA_Pairs_train.txt")
	m_path = os.path.join(root_path, "data/train/QAPairsByCategory/C1_Modality_train.txt")
	p_path = os.path.join(root_path, "data/train/QAPairsByCategory/C2_Plane_train.txt")
	o_path = os.path.join(root_path, "data/train/QAPairsByCategory/C3_Organ_train.txt")
	a_path = os.path.join(root_path, "data/train/QAPairsByCategory/C4_Abnormality_train.txt")

	questions = load_ques2set(path)
	m_ques = load_ques2set(m_path)
	p_ques = load_ques2set(p_path)
	o_ques = load_ques2set(o_path)
	a_ques = load_ques2set(a_path)

	sub_ques = set()
	sub_ques.update(m_ques, p_ques, o_ques, a_ques)
	outside_ques = questions.difference(sub_ques)
	return outside_ques


def check_len(filename):
	q_len_num = {}
	a_len_num = {}
	exclude_ls = ["<break>", "<START>", "<END>", "<UNKNOWN>"]
	# exclude_ls = ["<START>", "<END>"]
	with open(filename, "r") as f:
		for row in f:
			qa = row.rstrip().split("|")
			# words_q = qa[1].split()
			words_q = VQADataProvider.text_to_list(qa[1])
			words_q = [word for word in words_q if word not in exclude_ls]
			if(len(words_q) in q_len_num):
				q_len_num[len(words_q)] += 1
			else:
				q_len_num[len(words_q)] = 1

			# words_a = qa[2].split()
			words_a = VQADataProvider.text_to_list(qa[2])
			words_a = [word for word in words_a if word not in exclude_ls]
			if(len(words_a) in a_len_num):
				a_len_num[len(words_a)] += 1
			else:
				a_len_num[len(words_a)] = 1
	return q_len_num, a_len_num

def check_img_type(folder):
	import os
	img_ls = os.listdir(folder)
	print("image number: ", len(img_ls))
	jpg_ls = [img for img in img_ls if img.endswith(".jpg")]
	print("jpg image number: ", len(jpg_ls))

def check_unique_ans_num():
	path = os.path.join(root_path, "data/train/All_QA_Pairs_train.txt")
	m_path = os.path.join(root_path, "data/train/QAPairsByCategory/C1_Modality_train.txt")
	p_path = os.path.join(root_path, "data/train/QAPairsByCategory/C2_Plane_train.txt")
	o_path = os.path.join(root_path, "data/train/QAPairsByCategory/C3_Organ_train.txt")
	a_path = os.path.join(root_path, "data/train/QAPairsByCategory/C4_Abnormality_train.txt")

	_, _, all_ans = load_all2set(path)
	_, _, m_ans = load_all2set(m_path)
	_, _, p_ans = load_all2set(p_path)
	_, _, o_ans = load_all2set(o_path)
	_, _, a_ans = load_all2set(a_path)

	sub_ans = set()
	sub_ans.update(m_ans, p_ans, o_ans, a_ans)
	outside_ans = all_ans.difference(sub_ans)

	sub_ans2 = set()
	sub_ans2.update(m_ans, p_ans, o_ans)

	return outside_ans, sub_ans2

def check_train_val_imgId_overlap():
	train_path = os.path.join(root_path, "data/train/train_ImageIDs.txt")
	val_path = os.path.join(root_path, "data/val/val_ImageIDs.txt")
	train_img_ids = []
	with open(train_path, "r") as f:
		for row in f:
			train_img_ids.append(row.strip())
	val_img_ids = []
	with open(val_path, "r") as f:
		for row in f:
			val_img_ids.append(row.strip())
	print("train+val images number: ", len(train_img_ids) + len(val_img_ids))
	comb_ids = train_img_ids + val_img_ids
	comb_ids = set(comb_ids)
	print("unique total images number: ", len(comb_ids))


if __name__ == "__main__":
	filename = os.path.join(root_path, "data/train_val/QAPairsByCategory/C4_Abnormality_train_val.txt")
	_, _, ans_set = load_all2set(filename)
	unq_words = set()
	for ans in ans_set:
		words = VQADataProvider.text_to_list(ans)
		for w in words:
			unq_words.add(w)
	print(len(unq_words))
	print(list(unq_words)[:5])
	print("<END>" in unq_words)



	# filename = "/Users/leishi/Desktop/Internship/vqa2019/ImageClef-2019-VQA-Med-Training/QAPairsByCategory/C4_Abnormality_train.txt"
	# all_ans, all_img, all_ques, all_q_i_pairs, all_a_a_pairs = check_raw_data(filename)
	# print("\nunique answer length: ", len(all_ans))
	# print("\nunique img length: ", len(all_img))
	# print("\nunique question length: ", len(all_ques))
	# # print("\nall answer")
	# # print(all_ans)
	# print("\nunique question-image pairs: ", len(all_q_i_pairs))
	# print("\nunique question-answer pairs: ", len(all_a_a_pairs))
	# # print(all_ans)




	# as_m = ["XR - Plain Film", "CT - noncontrast", "CT w/contrast (IV)", "CT - GI & IV Contrast", "CTA - CT Angiography", 
	# "CT - GI Contrast", "CT - Myelogram", "Tomography", "MR - T1W w/Gadolinium", "MR - T1W - noncontrast", "MR - T2 weighted", 
	# "MR - FLAIR", "MR - T1W w/Gd (fat suppressed)", "MR T2* gradient,GRE,MPGR,SWAN,SWI", "MR - DWI Diffusion Weighted", 
	# "MRA - MR Angiography/Venography", "MR - Other Pulse Seq.", "MR - ADC Map (App Diff Coeff)", "MR - PDW Proton Density", 
	# "MR - STIR", "MR - FIESTA", "MR - FLAIR w/Gd", "MR - T1W SPGR", "MR - T2 FLAIR w/Contrast", "MR T2* gradient GRE", "US - Ultrasound", 
	# "US-D - Doppler Ultrasound", "Mammograph", "BAS - Barium Swallow", "UGI - Upper GI", "BE - Barium Enema", "SBFT - Small Bowel", 
	# "AN - Angiogram", "Venogram", "NM - Nuclear Medicine", "PET - Positron Emission"]

	# as_p = ["Axial", "Sagittal", "Coronal", "AP", "Lateral", "Frontal", "PA", "Transverse", "Oblique", "Longitudinal", "Decubitus", 
	# "3D Reconstruction", "Mammo - MLO", "Mammo - CC", "Mammo - Mag CC", "Mammo - XCC"]

	# as_o = ["Breast", "Skull and Contents", "Face, sinuses, and neck", "Spine and contents", "Musculoskeletal", "Heart and great vessels", 
	# "Lung, mediastinum, pleura", "Gastrointestinal", "Genitourinary", "Vascular and lymphatic"]

	# # as_all = as_m + as_p + as_o

	# as_m = [a.lower() for a in as_m]
	# as_p = [a.lower() for a in as_p]
	# as_o = [a.lower() for a in as_o]
	# # as_all = [a.lower() for a in as_all]

	# filename = "/home/leishi/vqa2019/data/train/QAPairsByCategory/C3_Organ_train.txt"

	# in_data_not_readme, in_readme_not_data = check_raw_readme(filename, as_o)
	# print("\ndifference between answer of raw data and answer in readme")
	# print(in_data_not_readme)
	# print("\ndifference between answer in readme and answer of raw data")
	# print(in_readme_not_data)




	# q_i_dic_for_ans = check_q_i_a_pair(filename)
	# num_list = []
	# for key in q_i_dic_for_ans:
	# 	num_list.append(len(q_i_dic_for_ans[key]))
	# # print("\n# of answers for each question-image pair")
	# print(max(num_list), min(num_list))

	# sep_answer_list, sep_answer_set = check_multi_answer(filename)
	# # print("# of separated answers for each question-image pair")
	# # print([len(a) for a in sep_answer_list])
	# x0 = [1 for a in sep_answer_list if len(a)>1]
	# print("\n# of q-i pairs that have multiple answers")
	# print(sum(x0))
	# x1 = [1 for b in sep_answer_set if len(b)>3]
	# print("\n# of unique answer which have more than one sub answers")
	# print(sum(x1))
	# print("answers that have multiple sub answers")
	# print([b for b in sep_answer_set if len(b)>3])

	# file1 = "/Users/leishi/Desktop/Internship/vqa2019/ImageClef-2019-VQA-Med-Training/QAPairsByCategory/C4_Abnormality_train.txt"
	# file2 = "/Users/leishi/Desktop/Internship/vqa2019/ImageClef-2019-VQA-Med-Training/QAPairsByCategory/C3_Organ_train.txt"
	# overlap = check_ques_overlap(file1, file2)
	# print("\n# of overlappling questions")
	# print(overlap)

	# diff_ques = check_ques_completeness()
	# print("# of uncovered questions: ", len(diff_ques))
	# print(diff_ques)

	# filename = "/home/leishi/vqa2019/data/train/QAPairsByCategory/C4_Abnormality_train.txt"
	# filename = "/home/leishi/vqa2019/data/train/All_QA_Pairs_train.txt"
	# q_len_num, a_len_num = check_len(filename)
	# print("\nquestion length")
	# # # for k, v in sorted(q_len_num.items(), key=lambda item: item[1]):
	# # # 	print("%d: %d" % (k, v))
	# for k in sorted(q_len_num.keys()):
	# 	print("%d: %d" % (k, q_len_num[k]))
	# print("\nanswer length")
	# # # for k, v in sorted(a_len_num.items(), key=lambda item: item[1]):
	# # # 	print("%d: %d" % (k, v))
	# for k in sorted(a_len_num.keys()):
	# 	print("%d: %d" % (k, a_len_num[k]))


	# # number of classes in each category (first 3 categories)
	# filename = "/home/leishi/vqa2019/data/train/QAPairsByCategory/C3_Organ_train.txt"
	# ans_set, _, _, _, _ = check_raw_data(filename)
	# print(len(ans_set))
	# print(ans_set)


	# folder = "/home/leishi/vqa2019/data/%s/%s_images" % ("val", "val")
	# check_img_type(folder)	

	# # check answer completeness
	# outside_ans, sub_ans2 = check_unique_ans_num()
	# print("\noutside answers")
	# print(outside_ans)
	# print("number of unique ansers in first 3 categories: ", len(sub_ans2))

	# # check if train and val data have overlap img ids
	# check_train_val_imgId_overlap()

