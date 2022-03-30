from multiprocessing.spawn import freeze_support
from os import listdir
from os.path import isfile, join
import pickle
from collections import Counter
import random, math
import math
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

def load_data(data_path):
    corpus = np.empty([1])
    with (open(data_path + 'corpus', 'rb')) as filehandle_2:
            corpus = pickle.load(filehandle_2)

    if corpus.shape[0] < 2:
        print("corpus not found")
        exit()

    print("corpus found")
    print("number of words in courpus: {}".format(len(corpus)))
    return corpus

def subsample_frequent_words(corpus):
    filtered_corpus = []
    print("getting word counts....")
    word_counts = dict(Counter(corpus))
    print("summing word counts....")
    sum_word_counts = sum(list(word_counts.values()))
    print("getting probabilities....")
    word_counts = {word: word_counts[word]/float(sum_word_counts) for word in word_counts}
    counter = 0
    print("filtering corpus...")
    for word in tqdm(corpus):
        if random.random() < (1+math.sqrt(word_counts[word] * 1e3)) * 1e-3/float(word_counts[word]):
            filtered_corpus.append(word)
        counter += 1
    print("number of words in filtered corpus: {}".format(len(filtered_corpus)))
    return np.array(filtered_corpus, dtype=np.uint32)

if __name__ == '__main__':
    data_path = 'data/'    
    output_path = 'data/'
    corpus = load_data(data_path)
    filtered_corpus = subsample_frequent_words(corpus)
    with open(output_path+'filtered_corpus', 'wb') as filehandle_1:
            pickle.dump(filtered_corpus, filehandle_1)
    print("done")