import json
import sys
import pickle
import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
from utils.data_provider import VQADataProvider
from gensim.models import KeyedVectors # load word2vec
import numpy as np


def make_vocab(sentence_ls, vocab_size=-1):
    word_fre_dic = {}
    for sent in sentence_ls:
        word_ls = VQADataProvider.text_to_list(sent)
        for word in word_ls:
            if(word in word_fre_dic):
                word_fre_dic[word] += 1
            else:
                word_fre_dic[word] = 1

    # sort
    vocab_ls = [k for (k, v) in sorted(word_fre_dic.items(), key= lambda x: x[1], reverse=True)]
    if(vocab_size != -1 and vocab_size <= len(vocab_ls)):
        vocab_ls = vocab_ls[:vocab_size]

    vocab_ls.reverse()
    # create dict with index
    vocab_dict = {}
    for i in range(len(vocab_ls)):
        vocab_dict[vocab_ls[i]] = i

    return vocab_dict

def make_vocab_ans(sentence_ls, vocab_size=-1):
    word_fre_dic = {}
    for sent in sentence_ls:
        word_ls = VQADataProvider.text_to_list(sent)
        for word in word_ls:
            if(word in word_fre_dic):
                word_fre_dic[word] += 1
            else:
                word_fre_dic[word] = 1

    # sort
    vocab_ls = [k for (k, v) in sorted(word_fre_dic.items(), key= lambda x: x[1], reverse=True)]
    if(vocab_size != -1 and vocab_size <= len(vocab_ls)):
        vocab_ls = vocab_ls[:vocab_size-1]
    # add <unknown>
    vocab_ls.append("<UNK>")
    # # add <ZERO> for padding
    # vocab_ls.insert(0, "<ZERO>")

    return vocab_ls


def make_vocab_files(opt, filename, ques_or_ans):
    save_path = os.path.join(root_path, "vocab/%s_vocab.json"%ques_or_ans)
    # load data
    if(ques_or_ans == "question"):
        _, sentence_ls, _ = VQADataProvider.load_raw_iqa(filename)
    elif(ques_or_ans == "answer"):
        _, _, sentence_ls = VQADataProvider.load_raw_iqa(filename)
    else:
        sentence_ls = None
    vocab_dict = make_vocab(sentence_ls)
    # save to json file
    with open(save_path, "w") as f:
        json.dump(vocab_dict, f)
    print("%s-%s vocabulary saved" % (filename, ques_or_ans))

def make_ans_vocab_file(opt, filename):
    save_path = os.path.join(root_path, "vocab/answer_vocab.pkl")
    # loada data
    _, _, sentence_ls = VQADataProvider.load_raw_iqa(filename)
    vocab_ls = make_vocab_ans(sentence_ls)
    with open(save_path, "wb") as f:
        pickle.dump(vocab_ls, f)

    #for debug 
    print(vocab_ls[:10])

    return len(vocab_ls)

def check_ans_vocab(filename):
    _, _, sentence_ls = VQADataProvider.load_raw_iqa(filename)
    word_fre_dict = {}
    for sent in sentence_ls:
        word_ls = VQADataProvider.text_to_list(sent)
        for word in word_ls:
            if(word in word_fre_dict):
                word_fre_dict[word] += 1
            else:
                word_fre_dict[word] = 1
    # sort
    word_fre_dict = sorted(word_fre_dict.items(), key=lambda kv: kv[1], reverse=True)
    return word_fre_dict


def make_vocab_for_embedding():
    embedding_path = os.path.join(root_path, "embedding/BioWordVec_PubMed_MIMICIII_d200.vec.bin")
    MED = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    # print(type(MED))
    word_dict = MED.wv.vocab
    # print(type(words))
    # print(len(words))
    raw_words = list(word_dict.keys())
    print(len(raw_words))

    # remove the word not appear in the dataset (training + val + test)
    # load all word
    all_word_in_data = set()
    train_val_path = os.path.join(root_path, "data/train_val/All_QA_Pairs_train_val.txt")
    test_path = os.path.join(root_path, "data/test/VQAMed2019_Test_Questions.txt")
    with open(train_val_path, "r") as f:
        print("process train_val file")
        for row in f:
            i_q_a = row.rstrip().split("|")
            q_words = VQADataProvider.text_to_list(i_q_a[1])
            q_words = q_words[:-1]
            for w in q_words:
                all_word_in_data.add(w)

    with open(test_path, "r") as f:
        print("process test file")
        for row in f:
            i_q = row.rstrip().split("|")
            q_words = VQADataProvider.text_to_list(i_q[1])
            q_words = q_words[:-1]
            for w in q_words:
                all_word_in_data.add(w)
    # filter
    words = []
    print("filter")
    for w in all_word_in_data:
        if w in word_dict:
            words.append(w)
    print(len(words))

    # add padding
    words.insert(0, "<PAD>")
    # # add UNK
    # words.append("<UNK>")
    # save words list to file for mapping questions to a list of indices
    save_path = os.path.join(root_path, "embedding/embed_mapping.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(words, f)

    # save the matrix of embedding to file
    save_path_2 = os.path.join(root_path, "embedding/embedding_matrix.npy")
    embedding_matrix = np.zeros((len(words), 200))
    for i in range(1, len(words)):
        embedding_matrix[i] = MED[words[i]]
    np.save(save_path_2, embedding_matrix)

    return words, embedding_matrix


    

if __name__ == "__main__":
    # words, embedding_matrix = make_vocab_for_embedding()
    # assert(len(words) == embedding_matrix.shape[0])
    # print(words[0])
    # print(embedding_matrix[0])
    # print(words[-1])
    # print(embedding_matrix[-1])
    # print(words)



    # test_ls = ["hello world, hello", "this that", "hello world"]
    # test_dict = make_vocab(test_ls)
    # print(test_dict)

    # # question vocabulary
    # train_data_path = os.path.join(root_path, "data/train_val/All_QA_Pairs_train_val.txt")
    # make_vocab_files(1, train_data_path, "question")

    # abnormality answer vocabulary
    train_data_path = os.path.join(root_path, "data/train_val/QAPairsByCategory/C4_Abnormality_train_val.txt")
    ans_voc_len = make_ans_vocab_file(1, train_data_path)
    # train_data_path = os.path.join(root_path, "data/train/QAPairsByCategory/C4_Abnormality_train.txt")
    # ans_voc_len = make_ans_vocab_file(1, train_data_path)
    print(ans_voc_len)

    # # test
    # with open(os.path.join(root_path, "vocab/answer_vocab.pkl"), "rb") as f:
    #     ans_voc_ls = pickle.load(f)
    # print(ans_voc_ls[:10])
    # print(ans_voc_ls[-1])
    # print(len(ans_voc_ls))

    # # check word frequency of answers
    # ans_file = os.path.join(root_path, "data/train/QAPairsByCategory/C4_Abnormality_train.txt")
    # word_fre_dict = check_ans_vocab(ans_file)
    # # print(word_fre_dict)
    # counter = 0
    # for (k,v) in word_fre_dict:
    #     if v>1:
    #         counter += 1
    # print(len(word_fre_dict))
    # print(counter)

