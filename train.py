import datetime
import time
import sys
import os
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(os.path.join(root_path, "utils"))
sys.path.append(os.path.join(root_path, "models"))
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision

# from models.coatt_model import mfh_coatt_Med
from models.coatt_model_emb import mfh_coatt_Med_emb
# from models.coatt_model_lstm import mfh_coatt_Med_lstm
import config
from utils.data_provider import VQADataProvider, VQADataset
from utils.validate import evaluate


# load hyperparameters
opt = config.parse_opt()

# set GPU device
torch.cuda.set_device(opt.TRAIN_GPU_ID)

# checkpoints to save model
model_checkpoints_folder = os.path.join(root_path, "models/checkpoints")

# provide data
train_data 	= VQADataset(mode="train", opt=opt)
train_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=opt.BATCH_SIZE, num_workers=1) 

# load the question word embedding matrix
embedding_matrix = np.load(os.path.join(root_path, "embedding/embedding_matrix.npy"))
embedding_matrix_tensor = torch.from_numpy(embedding_matrix).float()

# initialize model
model = mfh_coatt_Med_emb(opt, embedding_matrix_tensor)
# use GPU
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.INIT_LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=opt.DECAY_RATE)
# loss function for classification
weights = [1.0]*opt.ALL_CLASS_NUM
weights[-1] = 0.25
weights = torch.FloatTensor(weights).cuda()
loss_func1 = nn.CrossEntropyLoss(weight=weights, reduction="none")
# loss function for producting words of answer
loss_func2 = nn.KLDivLoss(reduction="none")

train_losses = []
for epoch in range(1, opt.EPOCH+1):
	model.train()
	print("\n%d epoch begins..." % epoch)
	scheduler.step()
	epoch_classification_loss = 0
	epoch_producing_loss = 0
	for step, (img_matrix, q_idx_ls, q_raw_category, q_category_gt, q_etm_topic, ans_raw_category, ans_category_gt, ans_vec_gt) in enumerate(train_loader):
		batch_size = img_matrix.size()[0]
		# move datat to GPU
		img_matrix = img_matrix.cuda().float()
		# print(q_idx_ls)
		q_idx_ls = q_idx_ls.cuda().long()
		q_category_gt = q_category_gt.cuda().float()
		q_etm_topic = q_etm_topic.cuda().float()
		ans_raw_category = ans_raw_category.cuda().long()
		# ans_category_gt = ans_category_gt.cuda().long()
		ans_vec_gt = ans_vec_gt.cuda().float()

		# print(img_matrix.size())
		# print(q_idx_ls.size())
		# print(q_category_gt.size())
		# print(q_etm_topic.size())

		optimizer.zero_grad()
		classification, ans_producing = model(img_matrix, q_idx_ls, q_category_gt, q_etm_topic, mode="train")
		# compute the loss
		loss1 = torch.sum(torch.mul((1 - q_category_gt[:, -1]), loss_func1(classification, ans_raw_category)))
		# print("model prediction size, ", ans_producing.size())
		# print("gt size, ", ans_vec_gt.size())
		# print("loss2 size, ", loss_func2(ans_producing, ans_vec_gt).size())
		# print("sum size, ", torch.sum(loss_func2(ans_producing, ans_vec_gt), dim=1).size())
		loss2 = torch.sum(torch.mul(q_category_gt[:, -1], torch.sum(loss_func2(ans_producing, ans_vec_gt), dim=[1, 2])))
		loss = opt.CLASSIFICATION_LOSS_WEIGHT*loss1 + loss2
		loss.backward()
		optimizer.step()
		# store the 2 losses separately
		epoch_classification_loss += (loss1.item())/batch_size
		epoch_producing_loss += (loss2.item())/batch_size

	train_losses.append((epoch_classification_loss, epoch_producing_loss))
	if(epoch%20 == 0):
		now = str(datetime.datetime.now())	
		print("{} Train Epoch{}: Classification Loss: {:.4f}; Producing Loss: {:.4f}".format(now, epoch, epoch_classification_loss, epoch_producing_loss))
		print("%d epoch ends \n" % epoch)

	if(epoch%50 == 0):
		# evaluate the model
		print("Validation starts ...")
		pred_class, _, acc, pred_ans, abnor_bleu = evaluate(model, opt, False, os.path.join(root_path, "results/predictions_%depoch.txt"%epoch))
		# print("\nconfusion matrix")
		# print(c_m)
		print("\naccuracy: ", acc)
		print("\nbleu for abnormality: ", abnor_bleu)
		# print("\nbleu for all answers: ", all_bleu)
		# save check point
		print("\nSaving checkpoints")
		model_save_path = os.path.join(model_checkpoints_folder, "%s-%d-iteration-model.pt"%(time.strftime("%Y%m%d%H"), epoch))
		torch.save(model.state_dict(), model_save_path)

		# for debug
		# oov_vec = model.oov.cpu().detach().numpy()
		# print(oov_vec)


print("\nTraining Finished")
print("\nClassification Loss and answer loss for each epoch")
print(train_losses)
print("\nFinal Result")
pred_save_path = os.path.join(root_path, "results/predictions_final.txt")
val_pred_class, val_c_m, val_acc, val_pred_ans, val_abnor_bleu = evaluate(model, opt, False, pred_save_path)
print("\n confusion matrix")
print(val_c_m)
print("\naccuracy: ", val_acc)
print("\nbleu for abnormality: ", val_abnor_bleu)
# print("\nbleu for all answers: ", val_all_bleu)

print("\nSaving the final model...")
model_save_path = os.path.join(root_path, "models/pretrained/%s_model.pt" % (time.strftime("%Y%m%d%H")))
torch.save(model.state_dict(), model_save_path)
print("Model saved at %s" % model_save_path)

# # for debug
# oov_vec = model.oov.cpu().detach().numpy()
# print(oov_vec)




