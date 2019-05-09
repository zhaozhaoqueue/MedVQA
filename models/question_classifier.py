import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from utils.data_provider import VQADataProvider

def train():
	# prepare the data
	# utils folder is in the root path, which is /home/leishi/vqa2019
	questions, labels = VQADataProvider.label_ques("train_val")
	vectorizer = TfidfVectorizer()
	ques_matrix = vectorizer.fit_transform(questions)

	# build classifier
	# svc
	classifier = SVC(gamma="scale", verbose=False)
	# # rf
	# classifier = RandomForestClassifier()
	# # gb
	# classifier = GradientBoostingClassifier()


	# train
	classifier.fit(ques_matrix, labels)

	# # training accuracy
	train_acc = classifier.score(ques_matrix, labels)
	print("training accuracy: ", train_acc)

	# save tf-idf
	with open(os.path.join(root_path, "models/pretrained/tfidf_train_val.pkl"), "wb") as f:
		pickle.dump(vectorizer, f)

	# save classifier
	with open(os.path.join(root_path, "models/pretrained/pretrained_svm_train_val.pkl"), "wb") as f:
		pickle.dump(classifier, f)

	# # debug
	# print("\nraw questions")
	# print(questions[:5])
	# print("question matrix")
	# print(vectorizer.transform(questions[:5]).todense())
	# # print("corresponding words of transformed questions")
	# # print(vectorizer.inverse_transform(vectorizer.transform(questions[:5])))
	# print("terms used")
	# print(vectorizer.get_feature_names())
	return vectorizer, classifier

# test load model
def test():
	# load tf-idf
	with open(os.path.join(root_path, "models/pretrained/tfidf.pkl"), "rb") as f:
		vectorizer = pickle.load(f)

	# load validation data
	val_ques, val_lab = VQADataProvider.label_ques("val")
	val_ques_matrix = vectorizer.transform(val_ques)

	# load model
	with open(os.path.join(root_path, "models/pretrained/pretrained_svm.pkl"), "rb") as f:
		clf = pickle.load(f)

	# compute validation accuracy
	val_acc = clf.score(val_ques_matrix, val_lab)
	print("validation accuracy")
	print(val_acc)

	# compute validation confusion matrix
	preds = clf.predict(val_ques_matrix)
	val_c_m = confusion_matrix(val_lab, preds)
	print("validation confusion matrix")
	print(val_c_m)

	# # for debug
	# print("\ntest on some validation question")
	# val_some_q = val_ques[5:10]
	# val_some_q_matrix = vectorizer.transform(val_some_q)
	# val_some_lab = val_lab[5:10]
	# print("question")
	# print(val_some_q)
	# print("prediction")
	# print(clf.predict(val_some_q_matrix))
	# print("ground truth")
	# print(val_some_lab)

# predict the category of a list of questions
def predict_q_category(questions):
	# load tf-idf
	with open(os.path.join(root_path, "models/pretrained/tfidf_train_val.pkl"), "rb") as f:
		vectorizer = pickle.load(f)
	# load model
	with open(os.path.join(root_path, "models/pretrained/pretrained_svm_train_val.pkl"), "rb") as f:
		clf = pickle.load(f)
	# create tf-idf matrix
	ques_matrix = vectorizer.transform(questions)
	# make predictions
	preds = clf.predict(ques_matrix)
	return preds


if __name__ == "__main__":
	train()
	# print("test")
	# test()