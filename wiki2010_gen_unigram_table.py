import math
import pickle
import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm

def create_unigram_tables(corpus):

    print("generating necessary arrays")
    word_count = Counter(corpus)

    print("generating the norm")
    power = 0.75
    norm = sum([math.pow(t, power) for t in  word_count.values()]) # Normalizing constant

    print("initializing the table")
    table_size = 2**30 # Length of the unigram table
    table = np.zeros(int(table_size), dtype=np.int32)

    print("generating unigrams...")
    p = 0 # Cumulative probability
    i = 0
    for j, unigram in enumerate(tqdm(corpus)):
        p += float(math.pow(word_count[unigram], power))/norm
        while i < table_size and float(i) / table_size < p:
            table[i] = j
            i += 1

    print("writing to file..")
    path = "data/"
    with open(path+'unigram_table_3', 'wb') as filehandle_2:
        pickle.dump(table, filehandle_2)
    
    del table

if __name__ == '__main__':  
    path = "data/"
    #file_paths = [f for f in listdir(path) if isfile(join(path, f))]
    #num_files = len(file_paths)
    #print("{} files found".format(num_files))
    corpus = []
    with (open(path+"filtered_corpus", 'rb')) as filehandle_1:
        corpus = pickle.load(filehandle_1)


    #unique = pickle.load(filehandle_2)

    #if (not corpus) or (not unique):
        #print("Missing files!")
        #exit()

    print("generating unigram tables...")
    create_unigram_tables(corpus)
    print("done!")