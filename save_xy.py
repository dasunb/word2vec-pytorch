import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import pickle
import ctypes
import math
import os
import pandas as pd
import gc

default_context_size = 4
split_val = 0.8
num_negative = 5

#Dataset path
path = "data/"

corpus = []
seq_len = default_context_size

print("Loading corpus...")
with (open(path+"filtered_corpus", 'rb')) as filehandle_1:
    corpus = pickle.load(filehandle_1)

corpus = corpus.astype(np.int32)
#corpus = np.array(corpus, dtype=np.int32)
print("Total corpus size: {}".format(corpus.shape[0]))
train_corpus = corpus[:math.floor(split_val*corpus.shape[0])]
test_corpus = corpus[math.floor(split_val*corpus.shape[0]):]

#FOR TESTING
#train_corpus = corpus[:10000000]

unique_words = list(set(corpus))
num_samples_train, num_unique_words_train = train_corpus.shape[0], len(unique_words)
data_len_train = num_samples_train - seq_len

data_summary = [['num_unique_words', len(unique_words)], ['context_size', default_context_size], ['negative sample size', num_negative], ['split train/test', split_val]]
df_data = pd.DataFrame(data_summary, columns=["attribute", "value"])
df_data.to_csv(path+'data_summary.csv', sep='\t', header='column_names')

shared_array_base_x_train = mp.Array(ctypes.c_int32,  data_len_train*seq_len)
shared_array_x_train = (np.ctypeslib.as_array(shared_array_base_x_train.get_obj())).reshape(data_len_train, seq_len)

shared_array_base_y_train = mp.Array(ctypes.c_int32, data_len_train)
shared_array_y_train = (np.ctypeslib.as_array(shared_array_base_y_train.get_obj())).reshape(data_len_train)


half_len = int(seq_len/2)
mask = np.ones(seq_len + 1, dtype=bool)
mask[half_len] = False

print("caching training data...")
for index in tqdm(range(data_len_train)):
    window = train_corpus[index:index + seq_len+1]
    shared_array_x_train[index] = window[mask]
    shared_array_y_train[index] = window[half_len]

print("Saving training data...")
with open(path+"x_train", 'wb') as output_datafile:
    pickle.dump(shared_array_x_train, output_datafile)

with open(path+"y_train", 'wb') as output_datafile:
    pickle.dump(shared_array_y_train, output_datafile)

del shared_array_x_train, shared_array_y_train
gc.collect()

num_samples_test, num_unique_words_test = test_corpus.shape[0], len(unique_words)
data_len_test = num_samples_test - seq_len

shared_array_base_x_test = mp.Array(ctypes.c_int32,  data_len_test*seq_len)
shared_array_x_test = (np.ctypeslib.as_array(shared_array_base_x_test.get_obj())).reshape(data_len_test, seq_len)

shared_array_base_y_test = mp.Array(ctypes.c_int32, data_len_test)
shared_array_y_test = (np.ctypeslib.as_array(shared_array_base_y_test.get_obj())).reshape(data_len_test)


half_len = int(seq_len/2)
mask = np.ones(seq_len + 1, dtype=bool)
mask[half_len] = False

print("caching test data...")
for index in tqdm(range(data_len_test)):
    window = test_corpus[index:index + seq_len+1]
    shared_array_x_test[index] = window[mask]
    shared_array_y_test[index] = window[half_len]

print("Saving test data...")
with open(path+"x_test", 'wb') as output_datafile:
    pickle.dump(shared_array_x_test, output_datafile)

with open(path+"y_test", 'wb') as output_datafile:
    pickle.dump(shared_array_y_test, output_datafile)

del shared_array_x_test, shared_array_y_test
gc.collect()