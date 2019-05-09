import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import os
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
import sys
sys.path.append(root_path)
import config
'''
Pre-trained Image NN
'''
# use resnet152
model_conv = torchvision.models.resnet152(pretrained=True)

# # use densenet 121
# model_conv = torchvision.models.densenet121(pretrained=True)


# # use ETM-tuned resnet152
# load_path = os.path.join(root_path, "models/pretrained/etm_tuned_resnet.pt")

# # use ETM-tuned DenseNet121
# load_path = os.path.join(root_path, "models/pretrained/tuned_densenet.model")

# model_conv = torch.load(load_path)
# model_conv.eval()

for param in model_conv.parameters():
    param.requires_grad = False             # just use the pre-trained network to extract features from images
# Parameters of newly constructed modules have requires_grad=True by default


class mfh_coatt_Med_lstm(nn.Module):
    def __init__(self, opt, embedding_matrix):
        super(mfh_coatt_Med_lstm, self).__init__()
        self.opt = opt
        self.batch_size = self.opt.BATCH_SIZE
        self.JOINT_EMB_SIZE = opt.MFB_FACTOR_NUM * opt.MFB_OUT_DIM  # mfh: a layer of mfb's.
        

        self.Embedding = nn.Embedding.from_pretrained(embedding_matrix)  # 200 is the dim of MED embedding
        self.Embedding.weight.requires_grad = False
        # for unknown word, its index should be -1
        self.oov = torch.nn.Parameter(data=torch.rand(1,200))
        self.oov_index = -1
        self.dim = 200


        # use bidirectinal LSTM, set bidirectional = True; when use bidirectional LSTM, change hidden_size to half of the original size
        self.LSTM = nn.LSTM(input_size=200, hidden_size=int(opt.LSTM_UNIT_NUM/2), num_layers=1, batch_first=False, bidirectional=True)  # 200 is the embedding dim

        self.Softmax = nn.Softmax(dim=1)

        self.Linear1_q_proj = nn.Linear(opt.LSTM_UNIT_NUM*opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear2_q_proj = nn.Linear(opt.LSTM_UNIT_NUM*opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear3_q_proj = nn.Linear(opt.LSTM_UNIT_NUM*opt.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)

        # resnet152
        self.img_feature_extr = nn.Sequential(*list(model_conv.children())[:-2])  # b_size x IMAGE_CHANNEL x IMAGE_WIDTH x IMAGE_WIDTH, IMAGE_CHANNEL=2048

        # fine tune last conv block of ResNet-152
        for param in self.img_feature_extr[7].parameters():
            param.requires_grad = True


        # # densenet121
        # self.img_feature_extr = nn.Sequential(*list(model_conv.children())[:-1])
        
        # # resnet152, remove the last conv block
        # self.img_feature_extr = nn.Sequential(*list(model_conv.children())[:-3])

        self.Conv1_i_proj = nn.Conv2d(opt.IMAGE_CHANNEL, self.JOINT_EMB_SIZE, 1)
        self.Linear2_i_proj = nn.Linear(opt.IMAGE_CHANNEL*opt.NUM_IMG_GLIMPSE, self.JOINT_EMB_SIZE)
        self.Linear3_i_proj = nn.Linear(opt.IMAGE_CHANNEL*opt.NUM_IMG_GLIMPSE, self.JOINT_EMB_SIZE)


        '''
        Add new feature (question avg attention and ETM topic result) to MFB, as well as MFH
        '''
        # question classification result
        self.Linear2_qc_proj = nn.Linear(opt.QUES_CLASS_NUM, self.JOINT_EMB_SIZE)
        self.Linear3_qc_proj = nn.Linear(opt.QUES_CLASS_NUM, self.JOINT_EMB_SIZE)
        # etm topic
        self.Linear2_etm_topic_proj = nn.Linear(opt.ETM_TOP_NUM, self.JOINT_EMB_SIZE)
        self.Linear3_etm_topic_proj = nn.Linear(opt.ETM_TOP_NUM, self.JOINT_EMB_SIZE)


        self.Dropout_L = nn.Dropout(p=opt.LSTM_DROPOUT_RATIO)
        self.Dropout_M = nn.Dropout(p=opt.MFB_DROPOUT_RATIO)

        self.Conv1_Qatt = nn.Conv2d(opt.LSTM_UNIT_NUM, 512, 1)  # (in_channels, out_channels, kernel_size)

        self.Conv2_Qatt = nn.Conv2d(512, opt.NUM_QUESTION_GLIMPSE, 1)

        self.Conv1_Iatt = nn.Conv2d(opt.MFB_OUT_DIM, 512, 1)

        self.Conv2_Iatt = nn.Conv2d(512, opt.NUM_IMG_GLIMPSE, 1)

        self.Classifier = nn.Linear(opt.MFB_OUT_DIM*2, opt.ALL_CLASS_NUM)

        # self.Linear_predict = nn.Linear(opt.MFB_OUT_DIM*2, opt.NUM_OUTPUT_UNITS*opt.MAX_WORDS_IN_ANSWER)    # output size: b_size x opt.NUM_OUTPUT_UNITS x opt.MAX_WORDS_IN_ANSWER

        # use LSTM to genereate answer for abnormality category
        self.ans_LSTM = nn.LSTM(input_size=opt.MFB_OUT_DIM*2, hidden_size=opt.NUM_OUTPUT_UNITS, num_layers=1, batch_first=False)
        # self.ans_LSTM = nn.LSTMCell(input_size=opt.MFB_OUT_DIM*2, hidden_size=opt.NUM_OUTPUT_UNITS)

    # since qvec is not used, remove this argument
    # add two more features (ETM topics and question classification)
    def forward(self, ivec, q_indices, qc, etm_topic, mode):
        # if mode == 'val' or mode == 'test':
        #     # self.batch_size = self.opt.VAL_BATCH_SIZE
        #     # load all validation data one time
        #     # later change it to batch type
        #     self.batch_size = q_MED_Matrix.size[0]
        # else:  # model == 'train'
        #     self.batch_size = self.opt.BATCH_SIZE
        self.batch_size = ivec.size()[0]


        mask = (q_indices==self.oov_index).long()
        # print(mask.size())
        # print(mask)
        mask_ = mask.unsqueeze(dim=2).float()
        # print(mask_.size())
        # print(mask_)
        embed =(1-mask_)*self.Embedding((1-mask)*q_indices) + mask_*(self.oov.expand((10,self.dim)))
        # print(embed.size())


        q_MED_Matrix = embed

        q_MED_Matrix = q_MED_Matrix.permute(1, 0, 2)                # type float, q_max_len x b_size x emb_size
        lstm1, _ = self.LSTM(q_MED_Matrix)                     # q_max_len x b_size x hidden_size
        lstm1_droped = self.Dropout_L(lstm1)                    # q_max_len x b_size x hidden_size
        lstm1_resh = lstm1_droped.permute(1, 2, 0)                     # b_size x hidden_size x q_max_len
        lstm1_resh2 = torch.unsqueeze(lstm1_resh, 3)              # b_size x hidden_size x q_max_len x 1

        '''
        Question Attention
        '''        
        qatt_conv1 = self.Conv1_Qatt(lstm1_resh2)                   # b_size x 512 x q_max_len x 1           ; 512 is the output dim of Conv1_Qatt layer
        qatt_relu = F.relu(qatt_conv1)
        qatt_conv2 = self.Conv2_Qatt(qatt_relu)                     # b_size x opt.NUM_QUESTION_GLIMPSE x q_max_len x 1;
        qatt_conv2 = qatt_conv2.view(self.batch_size*self.opt.NUM_QUESTION_GLIMPSE, -1)  # reshape
        # qatt_conv2 = qatt_conv2.view(-1, 200*1)  # reshape          # b_size*opt.NUM_QUESTION_GLIMPSE x 200
        qatt_softmax = self.Softmax(qatt_conv2)
        qatt_softmax = qatt_softmax.view(self.batch_size, self.opt.NUM_QUESTION_GLIMPSE, -1, 1)  # reshape
        # print(qatt_softmax.size())
        qatt_feature_list = []
        for i in range(self.opt.NUM_QUESTION_GLIMPSE):
            t_qatt_mask = qatt_softmax.narrow(1, i, 1)              # b_size x 1 x q_max_len x 1            ; narrow(dimension, start, length)
            t_qatt_mask = t_qatt_mask * lstm1_resh2                 # b_size x hidden_size x q_max_len x 1
            t_qatt_mask = torch.sum(t_qatt_mask, 2, keepdim=True)   # b_size x hidden_size x 1 x 1
            qatt_feature_list.append(t_qatt_mask)
        qatt_feature_concat = torch.cat(qatt_feature_list, 1)       # b_size x (hidden_size * NUM_QUESTION_GLIMPSE) x 1 x 1


        '''
        Extract Image Features with pre-trained NN
        '''
        img_feature = self.img_feature_extr(ivec)    # b_size x IMAGE_CHANNEL x IMAGE_WIDTH x IMAGE_WIDTH
        # print(img_feature.size())
        # print(model_conv)

        '''
        Image Attention with MFB
        '''
        q_feat_resh = torch.squeeze(qatt_feature_concat)                                # b_size x (hidden_size * NUM_QUESTION_GLIMPSE)
        iatt_q_proj = self.Linear1_q_proj(q_feat_resh)                                  # b_size x JOINT_EMB_SIZE
        iatt_q_resh = iatt_q_proj.view(self.batch_size, self.JOINT_EMB_SIZE, 1, 1)      # b_size x JOINT_EMB_SIZE x 1 x 1

        i_feat_resh = img_feature.view(self.batch_size, self.opt.IMAGE_CHANNEL, self.opt.IMG_FEAT_SIZE, 1)  # b_size x IMAGE_CHANNEL x IMG_FEAT_SIZE x 1
        iatt_i_conv = self.Conv1_i_proj(i_feat_resh)                                     # b_size x JOINT_EMB_SIZE x IMG_FEAT_SIZE x 1

        iatt_iq_eltwise = iatt_q_resh * iatt_i_conv                                     # b_size x JOINT_EMB_SIZE x IMG_FEAT_SIZE x 1
        iatt_iq_droped = self.Dropout_M(iatt_iq_eltwise)                                # b_size x JOINT_EMB_SIZE x IMG_FEAT_SIZE x 1
        iatt_iq_permute1 = iatt_iq_droped.permute(0, 2, 1, 3).contiguous()                 # b_size x IMG_FEAT_SIZE x JOINT_EMB_SIZE x 1
        iatt_iq_resh = iatt_iq_permute1.view(self.batch_size, self.opt.IMG_FEAT_SIZE, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)
        iatt_iq_sumpool = torch.sum(iatt_iq_resh, 3, keepdim=True)                      # b_size x IMG_FEAT_SIZE x MFB_OUT_DIM x 1
        iatt_iq_permute2 = iatt_iq_sumpool.permute(0, 2, 1, 3)                             # b_size x MFB_OUT_DIM x IMG_FEAT_SIZE x 1
        iatt_iq_sqrt = torch.sqrt(F.relu(iatt_iq_permute2)) - torch.sqrt(F.relu(-iatt_iq_permute2))
        iatt_iq_sqrt = iatt_iq_sqrt.view(self.batch_size, -1)                           # b_size x (MFB_OUT_DIM x IMG_FEAT_SIZE)
        iatt_iq_l2 = F.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(self.batch_size, self.opt.MFB_OUT_DIM, self.opt.IMG_FEAT_SIZE, 1)  # b_size x MFB_OUT_DIM x IMG_FEAT_SIZE x 1

        # 2 conv layers 1000 -> 512 -> 2
        iatt_conv1 = self.Conv1_Iatt(iatt_iq_l2)                    # b_size x 512 x IMG_FEAT_SIZE x 1
        iatt_relu = F.relu(iatt_conv1)
        iatt_conv2 = self.Conv2_Iatt(iatt_relu)                     # b_size x 2 x IMG_FEAT_SIZE x 1
        iatt_conv2 = iatt_conv2.view(self.batch_size*self.opt.NUM_IMG_GLIMPSE, -1)
        iatt_softmax = self.Softmax(iatt_conv2)
        iatt_softmax = iatt_softmax.view(self.batch_size, self.opt.NUM_IMG_GLIMPSE, -1, 1)
        iatt_feature_list = []
        for i in range(self.opt.NUM_IMG_GLIMPSE):
            t_iatt_mask = iatt_softmax.narrow(1, i, 1)              # b_size x 1 x IMG_FEAT_SIZE x 1
            t_iatt_mask = t_iatt_mask * i_feat_resh                 # b_size x IMAGE_CHANNEL x IMG_FEAT_SIZE x 1
            t_iatt_mask = torch.sum(t_iatt_mask, 2, keepdim=True)   # b_size x IMAGE_CHANNEL x 1 x 1
            iatt_feature_list.append(t_iatt_mask)
        iatt_feature_concat = torch.cat(iatt_feature_list, 1)       # b_size x (IMAGE_CHANNEL*2) x 1 x 1
        iatt_feature_concat = torch.squeeze(iatt_feature_concat)    # b_size x (IMAGE_CHANNEL*2)
        '''
        Fine-grained Image-Question MFH fusion
        '''

        mfb_q_o2_proj = self.Linear2_q_proj(q_feat_resh)               # b_size x 5000
        mfb_i_o2_proj = self.Linear2_i_proj(iatt_feature_concat)        # b_size x 5000
        mfb_iq_o2_eltwise = torch.mul(mfb_q_o2_proj, mfb_i_o2_proj)          # b_size x 5000
        # add two more features
        mfb_qc_o2_proj = self.Linear2_qc_proj(qc)
        mfb_iq_o2_eltwise = torch.mul(mfb_iq_o2_eltwise, mfb_qc_o2_proj)
        mfb_etm_o2_proj = self.Linear2_etm_topic_proj(etm_topic)
        mfb_iq_o2_eltwise = torch.mul(mfb_iq_o2_eltwise, mfb_etm_o2_proj)
        # done
        mfb_iq_o2_drop = self.Dropout_M(mfb_iq_o2_eltwise)
        mfb_iq_o2_resh = mfb_iq_o2_drop.view(self.batch_size, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)   # b_size x 1 x MFB_OUT_DIM x MFB_FACTOR_NUM
        mfb_iq_o2_sumpool = torch.sum(mfb_iq_o2_resh, 3, keepdim=True)    # b_size x 1 x MFB_OUT_DIM x 1
        mfb_o2_out = torch.squeeze(mfb_iq_o2_sumpool)                     # b_size x MFB_OUT_DIM
        mfb_o2_sign_sqrt = torch.sqrt(F.relu(mfb_o2_out)) - torch.sqrt(F.relu(-mfb_o2_out))
        mfb_o2_l2 = F.normalize(mfb_o2_sign_sqrt)

        mfb_q_o3_proj = self.Linear3_q_proj(q_feat_resh)               # b_size x 5000
        mfb_i_o3_proj = self.Linear3_i_proj(iatt_feature_concat)        # b_size x 5000
        mfb_iq_o3_eltwise = torch.mul(mfb_q_o3_proj, mfb_i_o3_proj)          # b_size x 5000
        # add two more features
        mfb_qc_o3_proj = self.Linear3_qc_proj(qc)
        mfb_iq_o3_eltwise = torch.mul(mfb_iq_o3_eltwise, mfb_qc_o3_proj)
        mfb_etm_o3_proj = self.Linear3_etm_topic_proj(etm_topic)
        mfb_iq_o3_eltwise = torch.mul(mfb_iq_o3_eltwise, mfb_etm_o3_proj)
        # done
        mfb_iq_o3_eltwise = torch.mul(mfb_iq_o3_eltwise, mfb_iq_o2_drop)
        mfb_iq_o3_drop = self.Dropout_M(mfb_iq_o3_eltwise)
        mfb_iq_o3_resh = mfb_iq_o3_drop.view(self.batch_size, 1, self.opt.MFB_OUT_DIM, self.opt.MFB_FACTOR_NUM)   # b_size x 1 x MFB_OUT_DIM x MFB_FACTOR_NUM
        mfb_iq_o3_sumpool = torch.sum(mfb_iq_o3_resh, 3, keepdim=True)    # b_size x 1 x MFB_OUT_DIM x 1
        mfb_o3_out = torch.squeeze(mfb_iq_o3_sumpool)                     # b_size x MFB_OUT_DIM
        mfb_o3_sign_sqrt = torch.sqrt(F.relu(mfb_o3_out)) - torch.sqrt(F.relu(-mfb_o3_out))
        mfb_o3_l2 = F.normalize(mfb_o3_sign_sqrt)

        mfb_o23_l2 = torch.cat((mfb_o2_l2, mfb_o3_l2), 1)               # b_size x (MFB_OUT_DIM * MFH_ORDER)
        '''
        two parallel submodel
        '''
        # classifier
        classification = self.Classifier(mfb_o23_l2)
        # because CrossEntropyLoss combines nn.LogSoftMax and nn.NLLLoss
        # classify_result = F.log_softmax(classification, dim=1)
        # print(classify_result.size())
        # print(classify_result[0, :])

        # use LSTM to generate words of answer
        # input_for_ans_lstm = mfb_o23_l2.view(self.batch_size, 5, -1)
        # input_for_ans_lstm = input_for_ans_lstm.permute(1, 0, 2)
        # print(input_for_ans_lstm.size())
        # lstm_ans, _ = self.ans_LSTM(input_for_ans_lstm)
        # lstm_ans_droped = self.Dropout_L(lstm_ans)
        # print(lstm_ans_droped.size())
        # lstm_ans_final = lstm_ans_droped.permute(1, 0, 2)
        # print(lstm_ans_final.size())
        # lstm_ans_final = F.log_softmax(lstm_ans_final, dim=2)

        input_for_ans_lstm = torch.unsqueeze(mfb_o23_l2, dim=0)
        # print(input_for_ans_lstm.size())
        # input_for_ans_lstm = mfb_o23_l2
        padding = torch.zeros(self.opt.MAX_WORDS_IN_ANSWER-1, self.batch_size, self.opt.MFB_OUT_DIM*2).cuda().float()
        input_for_ans_lstm = torch.cat((input_for_ans_lstm, padding), dim=0)
        # print(input_for_ans_lstm.size())
        h0 = torch.zeros(1, self.batch_size, self.opt.NUM_OUTPUT_UNITS).cuda().float()
        c0 = torch.zeros(1, self.batch_size, self.opt.NUM_OUTPUT_UNITS).cuda().float()
        output_from_ans_lstm, _ =  self.ans_LSTM(input_for_ans_lstm, (h0, c0))
        # print(output_from_ans_lstm.size())
        output_for_ans_generation = output_from_ans_lstm.permute(1, 0, 2)
        # print(output_for_ans_generation.size())
        prediction = F.log_softmax(output_for_ans_generation, dim=2)
        # print(prediction.size())
        # hx = torch.randn(self.batch_size, opt.NUM_OUTPUT_UNITS)
        # cx = torch.randn(self.batch_size, opt.NUM_OUTPUT_UNITS)
        # lstm_out = []
        # for i in range(opt.MAX_WORDS_IN_ANSWER):
        #     hx, cx = self.ans_LSTM(input_for_ans_lstm, (hx, cx))
        #     # print(hx.size())
        #     # print(cx.size())
        #     lstm_out.append(torch.unsqueeze(hx, dim=1))
        # lstm_ans_final = torch.cat(lstm_out, dim=1)
        # # print(lstm_ans_final.size())
        # lstm_ans_final = F.log_softmax(lstm_ans_final, dim=2)
        # # print(lstm_ans_final.size())

        return classification, prediction



if __name__ == "__main__":
    embedding_matrix = np.load(os.path.join(root_path, "embedding/embedding_matrix.npy"))
    embedding_matrix = torch.from_numpy(embedding_matrix).float()
    opt = config.parse_opt()
    model = mfh_coatt_Med_lstm(opt, embedding_matrix)
    model.cuda()
    img = torch.randn(8, 3, 224, 224).cuda().float()
    q = torch.randint(-1, 30, size=(8, 10)).cuda().long()
    # print(q)
    qc = torch.randn(8, 4).cuda().float()
    # qc = torch.randint(2, (8, 4), dtype=torch.DoubleTensor)
    etm_topic = torch.randn(8, 10).cuda().float()
    # etm_topic = torch.randint(2, (8, 20), dtype=torch.DoubleTensor)
    # pred = model(img, q, etm_topic, qc, mode="train")
    result = model(img, q, qc, etm_topic, mode="train")

