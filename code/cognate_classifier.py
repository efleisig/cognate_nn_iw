# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:41:56 2020

@author: evefl
"""

# First we load the tensors
"""
Lua conversion script:
 require 'torch'
> table_t = torch.totable(torch.load("FNAME.t7"))
> print(#table_t)
198
> for r=1,#table_t do
>> end
> out_file = io.open("FNAME.txt", "w")
> for r=1,#table_t do
>> for c=1,#table_t[1] do
>> out_file:write(table_t[r][c])
>> out_file:write("r")
>> end
>> out_file:write("c")
>> end
> io.close(out_file)
>


require 'torch'
function convert(fname)
table_t = torch.totable(torch.load(fname))
out_file = io.open(fname..".txt", "w")
for r=1, #table_t do
for c=1,#table_t[1] do
out_file:write(table_t[r][c])
out_file:write("r")
end
out_file:write("c")
end
io.close(out_file)
end
> convert("output_tensors_rg_all_5000.t7")
> convert("output_tensors_rg_all_10000.t7")
> convert("output_tensors_rg_all_15000.t7")
>

convert("output_tensors_rg_all_20000.t7"); convert("output_tensors_rg_all_25000.t7"); convert("output_tensors_rg_all_30000.t7"); convert("output_tensors_rg_all_35000.t7");convert("output_tensors_rg_all_40000.t7");convert("output_tensors_rg_all_45000.t7"); convert("output_tensors_rg_all_50000.t7");convert("output_tensors_rg_all_55000.t7")
convert("output_tensors_rg_all_20000.t7"); convert("output_tensors_rg_all_25000.t7"); convert("output_tensors_rg_all_30000.t7"); convert("output_tensors_rg_all_35000.t7");convert("output_tensors_rg_all_40000.t7");convert("output_tensors_rg_all_45000.t7"); convert("output_tensors_rg_all_50000.t7");convert("output_tensors_rg_all_55000.t7"); convert("output_tensors_rg_all_60000.t7"); convert("output_tensors_rg_all_65000.t7"); convert("output_tensors_rg_all_70000.t7"); convert("output_tensors_rg_all_75000.t7");convert("output_tensors_rg_all_80000.t7");convert("output_tensors_rg_all_85000.t7"); convert("output_tensors_rg_all_90000.t7");convert("output_tensors_rg_all_95000.t7"); convert("output_tensors_rg_all_100000.t7"); convert("output_tensors_rg_all_105000.t7"); convert("output_tensors_rg_all_110000.t7"); convert("output_tensors_rg_all_115000.t7");convert("output_tensors_rg_all_120000.t7");convert("output_tensors_rg_all_125000.t7"); convert("output_tensors_rg_all_130000.t7");convert("output_tensors_rg_all_135000.t7"); convert("output_tensors_rg_all_140000.t7"); convert("output_tensors_rg_all_145000.t7"); convert("output_tensors_rg_all_150000.t7"); convert("output_tensors_rg_all_155000.t7");convert("output_tensors_rg_all_160000.t7");convert("output_tensors_rg_all_165000.t7"); convert("output_tensors_rg_all_170000.t7")
"""

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
# PyTorch
#from torchvision import transforms, datasets, models
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
from sklearn import metrics
import pickle

import skimage.measure






"""Trains the neural network on the data.

Parameters
----------
all_x : Tensor
    Tensors of training data
y : Tensor
    Ground-truth tensors for training data
loss_fn : loss function

Returns
-------
model : torch.nn.module
    The trained model
"""
def train(all_x, y, loss_fn, model=None, name=""):
    
    # # Whether to train on a gpu
    # train_on_gpu = cuda.is_available()
    # print(f'Train on gpu: {train_on_gpu}')
    
    # # Number of gpus
    # if train_on_gpu:
    #     gpu_count = cuda.device_count()
    #     print(f'{gpu_count} gpus detected.')
    #     if gpu_count > 1:
    #         multi_gpu = True
    #     else:
    #         multi_gpu = False
        

    N, D_in, H, D_out = 1, all_x.size()[1], 100, 2
    
    print("N", N, "D_in", D_in, "H", H)
    n_classes = 2        
    
    if not model:
        model = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(H, n_classes))            # ,nn.LogSoftmax(dim=1)
    
    
    
    print("Begin training...", flush=True)
    
    learning_rate = 1e-4
    for t in range(600):
        #print(t, flush=True)
        
        total_loss = 0
        
        for x_index, x in enumerate(all_x):                                 #[:100]
                
            # Forward pass
            y_pred = model(x)
            #print(y_pred.size(), y.size())
            #print(x[:10], y[x_index], y_pred, flush=True)
            loss = loss_fn(y_pred, y[x_index])
            total_loss += loss
            
            
        
            # Backward pass
            # Zero the gradients before running the backward pass.
            model.zero_grad()
        
            # Backward pass
            loss.backward()
        
            # Update weights
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
        
        if t % 4 == 0:
            #print("Predicted", y_pred, "Actual", y[x_index], flush=True)
            print("Epoch ", t, ", loss:", loss.item(), flush=True)
            torch.save(model, "new_" + name + "_epoch_" + str(t) + "_loss_" + str(loss.item()) + ".pt")
        
    print("Reached end")
    return model

"""Tests the neural network on the data.

Parameters
----------
model : torch.nn.Module
all_x : Tensor
    Tensors of test data
y : Tensor
    Ground-truth tensors for test data
loss_fn : loss function
words : list, optional
    List of test examples corresponding to the items in all_x
    
Returns
-------
y_preds : Predicted classes for test data
"""

# Tests the input `model` on the features `all_x` and ground truth `y`.
def test(model, all_x, y, loss_fn, words=None):
    with torch.no_grad():
        
        total_loss = 0
        y_preds = []
        for x_index, x in enumerate(all_x):
                
            # Forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred, y[x_index])
            
            y_preds.append(y_pred)
            
            print("Predicted", y_pred, "Actual", y[x_index], flush=True)
        
            total_loss += loss
    
    # Evaluation
    print("Total loss:", total_loss)
    y_classes = [np.argmax(r) for r in y]
    yp_classes = [np.argmax(r) for r in y_preds]
    
    print_eval(y_classes, yp_classes, words)
    
    
    return y_preds
        
  
# Evaluates performance on Romance and Germanic subsets of the data.        
def analyze_subgroups(y, y_preds, y_words=None):
    
    
    
    y_classes = [np.argmax(r) for r in y]
    yp_classes = [np.argmax(r) for r in y_preds]
    
    rom_names = ['s', 'f', 'i', 'p', 'l']
    ger_names = ['d', 'n', 'e', 'a', 'w']
    
    
    
    y_ger = []
    yp_ger = []
    words_ger = []
    y_rom = []
    yp_rom = []
    words_rom = []
    y_cross = []
    yp_cross = []
    words_cross = []
    for index, entry in enumerate(y_words):
        words = entry.split()
        l1 = words[0]
        l2 = words[2]
        
        if l1 in rom_names and l2 in rom_names:
            y_rom.append(y_classes[index])
            yp_rom.append(yp_classes[index])
            if y_words:
                words_rom.append(y_words[index])
                
        elif l1 in ger_names and l2 in ger_names:
            y_ger.append(y_classes[index])
            yp_ger.append(yp_classes[index])
            if y_words:
                words_ger.append(y_words[index])
                
        else:
            y_cross.append(y_classes[index])
            yp_cross.append(yp_classes[index])
            if y_words:
                words_cross.append(y_words[index])
    
    print("ROMANCE RESULTS:")
    print_eval(y_rom, yp_rom, words_rom)
    print("GERMANIC RESULTS:")
    print_eval(y_ger, yp_ger, words_ger)
    print("CROSS RESULTS:")
    print_eval(y_cross, yp_cross, words_cross)
        

# Evaluates model performance given ground truth `y_classes` and predictions `yp_classes`.    
def print_eval(y_classes, yp_classes, words=None):
    
    if words:
        for y_index, y in enumerate(y_classes):
            if not y == yp_classes[y_index]:
                print("Erred on: ", words[y_index])
    
    cm = metrics.confusion_matrix(y_classes, yp_classes)
    print(cm, flush=True)
    f1 = metrics.f1_score(y_classes, yp_classes)
    print("F1:", f1)
    scores = metrics.precision_recall_fscore_support(y_classes, yp_classes)
    print(scores, flush=True)
    
    true_zeros = [item for item in y_classes if item==0]
    true_ones = [item for item in y_classes if item==1]
    print(len(true_zeros), len(true_ones))
    
    pred_zeros = [item for item in yp_classes if item==0]
    pred_ones = [item for item in yp_classes if item==1]
    print(len(pred_zeros), len(pred_ones))
    
    # Get accuracy
    num_correct = 0
    for index, item in enumerate(yp_classes):
        if (item==0 and y_classes[index]==0) or (item==1 and y_classes[index]==1):
            num_correct += 1
        # else:
        #     print(item, y_classes[index])
    
    print(num_correct, len(y_classes))
    accuracy = num_correct/len(y_classes)
    
    print("Accuracy:", accuracy)
    

# Convert text Lua tensors to correct dimensions for the input data
def load_lua_tensors(fname):
    
    text = [row for row in open(fname, "r")]
    
    for t in text:
        rows = t.split("c")
    
    for index, row in enumerate(rows):
        cols = row.split("r")
        rows[index] = [float(c) for c in cols[:-1]]
    
    rows = rows[:-1]
    print(len(rows), len(rows[0]), flush=True)
    
    # Flatten each pair of rows to get true dimension
    odds = rows[::2]
    evens = rows[1::2]
    
    for index, o in enumerate(odds):
        odds[index] += evens[index]
    
    x = np.array(odds)

    return x

# Make one-hot vectors before training.
# `old_char_to_id` is the existing character-index mapping, if needed.
def load_baseline(f_x, f_y, old_char_to_id=None):
    
    print("Loading baseline...")
    # Make concatenated one-hot vectors
    max_pt = 130 # expand if bigger
    cur_pt = 1
    

    char_to_id = {}
    id_to_char = {}
    one_hots = []
    for line in open(f_x, 'r', encoding="utf8"):
        
        line_one_hots = []
        
        for w_index, word in enumerate(line.split()):
            
            if w_index == 1 or w_index == 3:
                max_word_len = 30
            else:
                max_word_len = 1
        
            # Concatenate one-hot encodings for the characters in the word
            word_one_hots = [0]*(max_pt*max_word_len)
            for c_index, c in enumerate(word):
                
                if old_char_to_id:
                    index = old_char_to_id[c]
                else:
                    if c in char_to_id:
                        index = char_to_id[c]
                    else:
                        char_to_id[c] = cur_pt
                        id_to_char[cur_pt] = c
                        index = cur_pt
                        cur_pt += 1
                    
                #print(len(word_one_hots), c_index, max_pt, index)
                word_one_hots[c_index*max_pt + index] = 1
            
            
            line_one_hots += word_one_hots
            
        one_hots.append(line_one_hots)
            
    
    # Get ground truth data
    y = []
    for line in open(f_y, 'r', encoding="utf8"):
        if "1" in line:
            y.append([0, 1])
        else:
            y.append([1, 0])
    
    if not old_char_to_id:
        pickle.dump(char_to_id, open(f_x[:-4] + "_char_to_id.p", "wb"))
        pickle.dump(id_to_char, open(f_x[:-4] + "_id_to_char.p", "wb"))
    
    pickle.dump(one_hots, open(f_x[:-4] + "_onehot.p", "wb"))
    pickle.dump(y, open(f_y[:-4] + ".p", "wb"))
    
    x = torch.FloatTensor(one_hots)
    y = torch.FloatTensor(y)

    return x,y


# Converts text tensors from Lua code to numpy array
def load_char_vectors():
    
    x = pickle.load(open("rom_ger_train_onehot.p", "rb"))
    
    new_vals = np.zeros((68418, 3578)) #change 8190
    
    for tensor_set in range(5000, 195001, 5000):
        print("tensor_set", tensor_set, flush=True)
             
        if tensor_set != 95000 and tensor_set != 115000:
            tensor = load_lua_tensors("output_tensors_full_rg/text_versions/output_tensors_rg_all_" + str(tensor_set) + ".t7.txt")
    
    
            for entry in range(0, tensor.shape[0]-5, 5):
                index = int(tensor_set/5)-1000 + int(entry/5)
                
                #print("Entry number", index, flush=True)
                #entry=0
                cur_t = tensor[entry:entry+5]
                
                # if entry == 0:
                #print(cur_t.shape, flush=True)
                cur_t = cur_t.flatten()
                
                # Max pool to 3578
                cur_t = skimage.measure.block_reduce(cur_t, (100,), np.max)

                new_vals[index] = cur_t
    
                
        pickle.dump(new_vals, open("new_vals_through_"+ str(tensor_set) + ".p", "wb"))
        
    x = np.concatenate((x, new_vals), axis=1)  
    pickle.dump(new_vals, open("new_vals.p", "wb"))  


# Evaluates the performance of the baseline SVM.
def test_baseline(classifier, test_vectors, gt):
    predictions = classifier.predict(test_vectors)
    
    
    # Evaluation
    cm = metrics.confusion_matrix(gt, predictions)
    print(cm, flush=True)
    score = classifier.score(test_vectors, gt)
    print("Accuracy:", score, flush=True)
    f1 = metrics.f1_score(gt, predictions)
    print("F1:", f1)
    scores = metrics.precision_recall_fscore_support(gt, predictions)
    print(scores, flush=True)
    
    true_zeros = [item for item in gt if item==0]
    true_ones = [item for item in gt if item==1]
    print(len(true_zeros), len(true_ones))
    
    pred_zeros = [item for item in predictions if item==0]
    pred_ones = [item for item in predictions if item==1]
    print(len(pred_zeros), len(pred_ones))
    
    test_words = [line for line in open("rom_ger_test.txt", "r", encoding="utf8")]
    #analyze_subgroups(gt, predictions, test_words) 
    
    return predictions

# Creates feature vectors for the baseline SVM
def baseline_features(y):
    
    y_out = []
    l_order = "sfipldneawb"
    
    for index, line in enumerate(y):
        
        words = line.split()
        w1 = words[1]
        w2 = words[3]
        
        maxlen = max(len(w1), len(w2))
        c1 = np.array([ord(l) for l in w1])
        c1 = np.concatenate([c1, np.zeros(30-len(c1))])
        c2 = np.array([ord(l) for l in w2])
        c2 = np.concatenate([c2, np.zeros(30-len(c2))])
        diff = np.sum(c1==c2)

        y_out.append(diff)

    return np.array(y_out).reshape(-1, 1) 
        
   
# Runs the baseline SVM    
def run_svm():
    yb_train = [row.index(1) for row in pickle.load(open("basque_train_gt.p", "rb"))]
    yb_test = [row.index(1) for row in pickle.load(open("basque_test_gt.p", "rb"))]
    yrg_train = [row.index(1) for row in pickle.load(open("rom_ger_train_gt.p", "rb"))]
    yrg_test = [row.index(1) for row in pickle.load(open("rom_ger_test_gt.p", "rb"))]
    
    basque_train_words = [line for line in open("basque_train.txt", "r", encoding="utf8")]
    basque_test_words = [line for line in open("basque_test.txt", "r", encoding="utf8")]
    rg_train_words = [line for line in open("rom_ger_train.txt", "r", encoding="utf8")]
    rg_test_words = [line for line in open("rom_ger_test.txt", "r", encoding="utf8")]
    
    xb_train = baseline_features(basque_train_words)
    xb_test = baseline_features(basque_test_words)
    xrg_train = baseline_features(rg_train_words)
    xrg_test = baseline_features(rg_test_words)
    
    
    print(xrg_test[:10], rg_test_words[:10], yrg_test[:10])
    print(len(rg_train_words), len(xrg_train))
    svc_baseline = SVC()
    svc_baseline.fit(xrg_train, yrg_train)
    
    print("ROMANCE AND GERMANIC BASELINE RESULTS---------------------------------8", flush=True)
    test_baseline(svc_baseline, xrg_test, yrg_test)
    
    
    print("BASQUE BASELINE, NO FINE TUNING---------------------------------------", flush=True)
    test_baseline(svc_baseline, xb_test, yb_test)
    
    
    print("BASQUE BASELINE, WITH BASQUE ACCESS-----------------------------------", flush=True)
    x_both_train = np.concatenate((xrg_train, xb_train), axis=0)
    y_both_train = np.concatenate((yrg_train, yb_train), axis=0)
    svc_both_baseline = SVC()
    svc_both_baseline.fit(x_both_train, y_both_train)
    test_baseline(svc_both_baseline, xb_test, yb_test)
    
    
    print("BASQUE BASELINE, BASQUE ONLY------------------------------------------", flush=True)
    svc_b_baseline = SVC()
    svc_b_baseline.fit(xb_train, yb_train)
    test_baseline(svc_b_baseline, xb_test, yb_test)
    

loss_fn = torch.nn.MSELoss(reduction='sum')

#run_svm()

#load_char_vectors()


# print("RG SUBGROUP RESULTS---------------------------------------------------", flush=True)
# y_test = torch.FloatTensor(pickle.load(open("rom_ger_test_gt.p", "rb")))
# y_test_pred = [line for line in pickle.load(open("fixed_new_rg_256_test_predictions.p", "rb"))]
# test_words = [line for line in open("rom_ger_test.txt", "r")]
# analyze_subgroups(y_test, y_test_pred, test_words)    


old_char_to_id = (pickle.load(open("char_to_id.p", "rb")))
# x_test,y_test = load_baseline("rom_ger_test.txt", "rom_ger_test_gt.txt", old_char_to_id)
xb_test, yb_test = load_baseline("correct_basque_test.txt", "correct_basque_test_gt.txt", old_char_to_id)
xb_val, yb_val = load_baseline("correct_basque_val.txt", "correct_basque_val_gt.txt", old_char_to_id)
xb_train, yb_train = load_baseline("correct_basque_train.txt", "correct_basque_train_gt.txt", old_char_to_id)

# pickle.dump(xb_test, open("correct_basque_test_onehot.p", "wb"))
# pickle.dump(xb_val, open("correct_basque_val_onehot.p", "wb"))
# pickle.dump(xb_train, open("correct_basque_train_onehot.p", "wb"))
# pickle.dump(x_test, open("fixed_rg_test_onehot.p", "wb"))

xb_test = np.concatenate((xb_test, np.zeros((xb_test.shape[0], 716))), axis=1)
xb_val = np.concatenate((xb_val, np.zeros((xb_val.shape[0], 716))), axis=1)
xb_train = np.concatenate((xb_train, np.zeros((xb_train.shape[0], 716))), axis=1)
# x_test = np.concatenate((x_test, np.zeros((x_test.shape[0], 716))), axis=1)

pickle.dump(xb_test, open("correct_new_basque_test_onehot.p", "wb"))
pickle.dump(xb_val, open("correct_new_basque_val_onehot.p", "wb"))
pickle.dump(xb_train, open("correct_new_basque_train_onehot.p", "wb"))
# pickle.dump(x_test, open("fixed_new_rg_test_onehot.p", "wb"))


# Get RG results
model = torch.load("output_models_new/new_2_rg_epoch_256_loss_0.004484307952225208.pt")
print("ROMANCE AND GERMANIC RESULTS------------------------------------------")
rg_test_words = [line for line in open("rom_ger_test.txt", "r")]
y_rg_test_pred = test(model, torch.FloatTensor(x_test), torch.FloatTensor(y_test), loss_fn, rg_test_words)
pickle.dump(y_rg_test_pred, open("fixed_new_rg_256_test_predictions.p", "wb"))

# Get Basque results without fine-tuning
basque_test_words = [line for line in open("correct_basque_test.txt", "r")]
print("BASQUE RESULTS, NO FINE-TUNING----------------------------------------")
y_b_test_pred = test(model, torch.FloatTensor(xb_test), torch.FloatTensor(yb_test), loss_fn, basque_test_words)
pickle.dump(y_b_test_pred, open("correct_new_b_256_test_predictions_no_ft.p", "wb"))

# Basque with fine-tuning
print("BASQUE RESULTS, WITH FINE-TUNING--------------------------------------")
basque_model = train(torch.FloatTensor(xb_train), torch.FloatTensor(yb_train), loss_fn, model, name="correct_b_ft")
torch.save(basque_model, "correct_basque_fine_tuned_model.pt")
y_b_ft_test_pred = test(basque_model, torch.FloatTensor(xb_test), torch.FloatTensor(yb_test), loss_fn, basque_test_words)
pickle.dump(y_b_ft_test_pred, open("correct_new_b_256_test_predictions_ft.p", "wb"))

# Basque without pretraining
print("BASQUE RESULTS, NO PRETRAINING--------------------------------------")

basque_no_pt_model = train(torch.FloatTensor(xb_train), torch.FloatTensor(yb_train), loss_fn, name = "correct_b_no_pt")
torch.save(basque_no_pt_model, "correct_basque_no_pretraining_model.pt")
basque_no_pt_model = torch.load("correct_basque_no_pretraining_model.pt")
y_b_no_pt_test_pred = test(basque_no_pt_model, torch.FloatTensor(xb_test), torch.FloatTensor(yb_test), loss_fn, basque_test_words)
pickle.dump(y_b_no_pt_test_pred, open("correct_new_b_256_test_predictions_no_pretrain.p", "wb"))





