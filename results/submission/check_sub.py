org_file = "/Users/leishi/Desktop/Internship/vqa2019/data/test/VQAMed2019_Test_Questions.txt"
sub_file = "/Users/leishi/Desktop/Internship/vqa2019/results/submission/042907-submission_train_with_train_val.txt"

org_img = []
sub_img = []
with open(org_file, "r") as f:
	for row in f:
		qi = row.strip().split("|")
		org_img.append(qi[0])

with open(sub_file, "r") as f:
	for row in f:
		qi = row.strip().split("|")
		sub_img.append(qi[0])

assert len(org_img) == len(sub_img)
print(org_img == sub_img)
assert org_img == sub_img
