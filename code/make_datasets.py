# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:12:30 2020

@author: evefl
"""
import csv
from builtins import str
import pickle
import numpy as np
import random
from sklearn import model_selection
import copy


# Creates Romance-Germanic dataset from Etymological WordNet data
def make_datasets():
    fname = "data/etymwn.tsv"
    ctr = 0
    
    rom_text = []
    rom_ger_text = []
    
    rom_list = ['spa', 'fra', 'ita', 'por', 'lat']
    rom_ger_list = ['spa', "fra", "ita", "por", "lat", "deu", "nld", "eng", "dan", "swe"] #s f i p l d n e a w
    
    words_by_language = {'spa':[], "fra":[], "ita":[], "por":[], "lat":[], "deu":[], "nld":[], "eng":[], "dan":[], "swe":[]}
    
    with open(fname, 'r', encoding='utf-8') as tsvfile:
        for row in tsvfile:
            
            
            items = row.split("\t")

            l1 = items[0][:3].strip()
            w1 = items[0][4:].strip()
            l2 = items[2][:3].strip()
            w2 = items[2][4:-1].strip()
            rel = items[1]
            
            if len(w1.split())==1 and len(w2.split())==1:
                
                # Create datasets
                if l1 != l2:
                    if l1 in rom_ger_list and l2 in rom_ger_list:
                        
                        if l1 in rom_list and l2 in rom_list:
                        
                            rom_text.append([" ".join([l1, w1, l2, w2, "&"]), 1])
                        
                        words_by_language[l1].append(w1)
                        words_by_language[l2].append(w2)
                        
                        
                        rom_ger_text.append([" ".join([l1, w1, l2, w2, "&"]), 1])
        
        save_file("rom_ger_cognates_only.txt", [line[0] for line in rom_ger_text])
        

        # Add non-cognate data
        orig_rom_ger_text = copy.deepcopy(rom_ger_text)
        for index, line in enumerate(orig_rom_ger_text):
            words = line[0].split()
           
            # Make non-cognate example with same w1 word half of the time, same w2 word the other half of the time
            if index % 2 == 0:
                l1 = words[0]
                new_w1 = w1
                while is_cognate(new_w1, w2, rom_ger_text):
                    new_w1 = random.choice(words_by_language[l1])
               
               
                words[1] = new_w1
                rom_ger_text.append([" ".join(words), 0])
           
            else:
                l2 = words[2]
                new_w2 = w2
                while is_cognate(w1, new_w2, rom_ger_text):
                    #print(words)
                    new_w2 = random.choice(words_by_language[l2])
               
               
                words[3] = new_w2
                rom_ger_text.append([" ".join(words), 0])
            
            if index % 1000 == 0:
                print(index)
        
        
        # Replace language codes with single-character codes
        lang_chars = {'spa':'s', "fra":'f', "ita":'i', "por":'p', "lat":'l', 
                      "deu":'d', "nld":'n', "eng":'e', "dan":'a', "swe":'w'}
        for index, line in enumerate(rom_ger_text):
            if index ==5:
                print(words)
            words = line[0].split()
            words[0] = lang_chars[words[0]]
            words[2] = lang_chars[words[2]]
            rom_ger_text[index][0] = " ".join(words)
            if index ==5:
                print(rom_ger_text[:10])
        
        print("shuffling")
        # Shuffle data and make train/val/test files
        random.Random(4).shuffle(rom_ger_text)
            
        rom_text = [line for line in rom_ger_text if line[0].split()[0] in rom_list and line[0].split()[2] in rom_list]        

        X_rg = [line[0] for line in rom_ger_text]
        y_rg = [line[1] for line in rom_ger_text]
        
        X_r = [line[0] for line in rom_text]
        y_r = [line[1] for line in rom_text]
        
        X_r_train, X_r_test, y_r_train, y_r_test = model_selection.train_test_split(X_r, y_r, test_size=0.2, random_state=1)
        X_r_train, X_r_val, y_r_train, y_r_val = model_selection.train_test_split(X_r_train, y_r_train, test_size=0.2, random_state=1)
        
        X_rg_train, X_rg_test, y_rg_train, y_rg_test = model_selection.train_test_split(X_rg, y_rg, test_size=0.2, random_state=1)
        X_rg_train, X_rg_val, y_rg_train, y_rg_val = model_selection.train_test_split(X_rg_train, y_rg_train, test_size=0.2, random_state=1)
        
        X_rg_cognates_only = [line for index, line in enumerate(X_rg_train) if y_rg_train[index]==1]
        X_r_cognates_only = [line for index, line in enumerate(X_r_train) if y_r_train[index]==1]
        
        X_rg_cognates_only_val = [line for index, line in enumerate(X_rg_val) if y_rg_val[index]==1]
        X_rg_cognates_only_test = [line for index, line in enumerate(X_rg_test) if y_rg_test[index]==1]
       
        # Save data in files
        save_file("rom_ger_train_cognates_only.txt", X_rg_cognates_only)
        save_file("rom_ger_train.txt", X_rg_train)
        save_file("rom_ger_train_gt.txt", y_rg_train)
        save_file("rom_ger_val.txt", X_rg_val)
        save_file("rom_ger_val_gt.txt", y_rg_val)
        save_file("rom_ger_test.txt", X_rg_test)
        save_file("rom_ger_test_gt.txt", y_rg_test)
        
        
        save_file("rom_train_cognates_only.txt", X_r_cognates_only)
        save_file("rom_train.txt", X_r_train)
        save_file("rom_train_gt.txt", y_r_train)
        save_file("rom_val.txt", X_r_val)
        save_file("rom_val_gt.txt", y_r_val)
        save_file("rom_test.txt", X_r_test)
        save_file("rom_test_gt.txt", y_r_test)
        
        pickle.dump(rom_ger_text, open("rom_ger.p", "wb" ))
        pickle.dump(rom_text, open("rom.p", "wb" ))


# Creates the Basque-Spanish dataset
def make_basque_dataset():     

    basque_data = []
    s_words = []
    b_words = []
    for row in open("basque_spanish_cognates_only.txt", "r"):
        words = row.split()   
        print("w", words)
        b_words.append(words[0])
        s_words.append(words[1])
        basque_data.append([" ".join(["b", words[0], "s", words[1], "&"]), 1])
        
    
    # Add non-cognate data
    orig_rows = copy.deepcopy(basque_data)
    for index, line in enumerate(orig_rows):
        words = line[0].split()
       
        # Make non-cognate example with same w1 word half of the time and same w2 word the other half of the time
        if index % 2 == 0:
            new_b = random.choice(b_words)
            words[1] = new_b
            basque_data.append([" ".join(words), 0])
       
        else:
            new_s = random.choice(s_words)
            words[3] = new_s
            basque_data.append([" ".join(words), 0])
           
    # Shuffle data and make train/val/test files
    random.Random(4).shuffle(basque_data)   

    X_b = [line[0] for line in basque_data]
    y_b = [line[1] for line in basque_data]
    
    
    X_b_train, X_b_test, y_b_train, y_b_test = model_selection.train_test_split(X_b, y_b, test_size=0.25, random_state=1)
    X_b_train, X_b_val, y_b_train, y_b_val = model_selection.train_test_split(X_b_train, y_b_train, test_size=0.2, random_state=1)
    
   
    # Save data in files
    save_file("correct_basque_train.txt", X_b_train)
    save_file("correct_basque_train_gt.txt", y_b_train)
    save_file("correct_basque_val.txt", X_b_val)
    save_file("correct_basque_val_gt.txt", y_b_val)
    save_file("correct_basque_test.txt", X_b_test)
    save_file("correct_basque_test_gt.txt", y_b_test)
    
    pickle.dump(basque_data, open("correct_basque_all_data.p", "wb" ))
        
        
          
# Saves the file
def save_file(fname, data):
    
    with open(fname, "w", encoding='utf-8') as f:
        for line in data:
            f.write(str(line) + "\n")
                    
# Checks if words `w1` and `w2` are cognates
def is_cognate(w1, w2, text):           
    
    for line in text:
        #print(line[0])
        if w1 in line[0] and w2 in line[0]:
            return True
    return False


# Prints the number of cognates per language pair in `fname`.
def get_stats(fname):
    
    lang_list = ["spa", "fra", "ita", "por", "lat", "deu", "nld", "eng", "dan", "swe"]
    # Change if abbreviated as ['s','f', 'i', 'p', 'l', 'd', 'n', 'e', 'a', 'w']
    
    langs_dict = {}
    with open(fname, 'r', encoding='utf-8') as cogfile:
        for row in cogfile:
            
            items = row.split()
            
            l1 = items[0]
            l2 = items[2]
            
            if l1 not in langs_dict and l2 not in langs_dict:
                langs_dict[l1] = {l2: 1}
                langs_dict[l2] = {l1: 1}
            
            elif l1 not in langs_dict:
                langs_dict[l1] = {l2: 1}
                langs_dict[l2][l1] = 1
            
            
            elif l2 not in langs_dict:
                langs_dict[l2] = {l1: 1}
                langs_dict[l1][l2] = 1
            
            else:
                if l2 in langs_dict[l1]:
                    langs_dict[l1][l2] += 1
                else:
                    langs_dict[l1][l2] = 1
                    
                if l1 in langs_dict[l2]:
                    langs_dict[l2][l1] += 1
                else:
                    langs_dict[l2][l1] = 1
    
    mtx = []
    for l1 in lang_list:
        row = []
        for l2 in lang_list:
            if l2 in langs_dict[l1] and l1 != l2:
                row.append(langs_dict[l1][l2])
            elif l1 == l2:
                row.append("-")
            else:
                row.append(0)

        mtx.append(row)
            
    row_format ="{:>7}" * (len(lang_list) + 1)
    print(row_format.format("", *lang_list))
    for lang, row in zip(lang_list, mtx):
        print(row_format.format(lang, *row))
        
     
# Prints statistics and word pairs in the Etymological WordNet data.
def check_etymology():
    
    fname = "data/etymwn.tsv"
    ctr = 0
    
    langs_dict = {}
    with open(fname, 'r', encoding='utf-8') as tsvfile:
        for row in tsvfile:
            
            
            items = row.split("\t")
            #print(items)
            
            l1 = items[0][:3]
            l2 = items[2][:3]
            
            rel = items[1]
            
            if l1 not in langs_dict and l2 not in langs_dict:
                langs_dict[l1] = {l2: 1}
                langs_dict[l2] = {l1: 1}
            
            elif l1 not in langs_dict:
                langs_dict[l1] = {l2: 1}
                langs_dict[l2][l1] = 1
            
            
            elif l2 not in langs_dict:
                langs_dict[l2] = {l1: 1}
                langs_dict[l1][l2] = 1
            
            else:
                if l2 in langs_dict[l1]:
                    langs_dict[l1][l2] += 1
                else:
                    langs_dict[l1][l2] = 1
                    

                if l1 in langs_dict[l2]:
                    langs_dict[l2][l1] += 1
                else:
                    langs_dict[l2][l1] = 1

        
        lang_list = ['spa', "fra", "ita", "por", "lat", 
                     "deu", "nld", "eng", "dan", "swe"]

        for lang in lang_list:
            print(lang, langs_dict[lang])

        mtx = []
        
        for l1 in lang_list:
            row = []
            for l2 in lang_list:
                if l2 in langs_dict[l1] and l1 != l2:
                    row.append(langs_dict[l1][l2])
                elif l1 == l2:
                    row.append("-")
                else:
                    row.append(0)

            
            mtx.append(row)
        
        row_format ="{:>7}" * (len(lang_list) + 1)
        print(row_format.format("", *lang_list))
        for lang, row in zip(lang_list, mtx):
            print(row_format.format(lang, *row))



check_etymology()
make_datasets()
get_stats("rom_ger_train.txt")
get_stats("rom_ger_train_cognates_only.txt")
get_stats("rom_ger_test.txt")
get_stats("rom_train.txt")
get_stats("rom_test.txt")
get_stats("rom_ger_cognates_only.txt")

with open("rom_ger_val.txt", 'r', encoding='utf-8') as val_f:
    X_rg_val = [line.strip() for line in val_f]
    
with open("rom_ger_val_gt.txt", 'r', encoding='utf-8') as val_gt_f:    
    y_rg_val = [int(line.strip()) for line in val_gt_f]
    
X_rg_cognates_only_val = [line for index, line in enumerate(X_rg_val) if y_rg_val[index]==1]
print(X_rg_val, y_rg_val, X_rg_cognates_only_val)
#save_file("rom_ger_val_cognates_only.txt", X_rg_cognates_only_val)

# Make sub-validation set for LM training
with open("rom_ger_train_cognates_only.txt", 'r', encoding='utf-8') as val_f:
    X_rg_train_cog = [line.strip() for line in val_f]
    
X_rg_train_cog, X_rg_val_cog = model_selection.train_test_split(X_rg_train_cog, test_size=0.2, random_state=1)
#save_file("rom_ger_train_cognates_only.txt", X_rg_train_cog)
#save_file("rom_ger_val_cognates_only.txt", X_rg_val_cog)


make_basque_dataset()
get_stats("rom_ger_cognates_only.txt")           
