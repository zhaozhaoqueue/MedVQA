import os 
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
import pickle
# from data_explore import load_all2set

def create_mapping_file(mode):
	file_parent_path = os.path.join(root_path, "data/%s/QAPairsByCategory"%mode)
	file_ls = ["C1_Modality_%s.txt"%mode, "C2_Plane_%s.txt"%mode, "C3_Organ_%s.txt"%mode]
	target_classes = set()
	for file in file_ls:
		file_path = os.path.join(file_parent_path, file)
		_, _, ans_set = load_all2set(file_path)
		ans_set = set([item.lower() for item in ans_set])
		target_classes = target_classes|ans_set

	target_classes = list(target_classes)
	# save the mapping to file
	mapping_file_path = os.path.join(root_path, "utils/mapping.pkl")
	with open(mapping_file_path, "wb") as f:
		pickle.dump(target_classes, f)
	return target_classes

def load_mapping_ls():
	mapping_file_path = os.path.join(root_path, "utils/mapping.pkl")
	with open(mapping_file_path, "rb") as f:
		target_classes = pickle.load(f)

	return target_classes

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

if __name__ == "__main__":
	target_classes = create_mapping_file("train_val")
	print(len(target_classes))
	print(target_classes)
	
	# test loading
	target_classes = load_mapping_ls()
	print(len(target_classes))
	print(target_classes)