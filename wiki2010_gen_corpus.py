from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
from numpy.lib.arraysetops import unique
from tqdm import tqdm

dataset_path = "raw/"
file_paths = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

if not file_paths:
    print("No data")
    exit()

output_path = "data/"

num_files = len(file_paths)
print("{} files found".format(num_files))

file_counter = 0
all_words = []
for file_path in tqdm(file_paths):
    words = []
    if not file_path.split("_")[0] == "englishEtiquetado":
        continue

    #print("reading file {} of {}...".format(file_counter, num_files))
    full_path = dataset_path + file_path    
    lines = []
    with (open(full_path, 'rb')) as datafile_handle:
        lines = datafile_handle.readlines()

    for line in lines:
        line_str = str(line)
        if not line_str:
            continue
        line_arr = line_str[2:].split()
        if len(line_arr) < 4:
            continue
        if line_arr[0][0] == "<":
            continue
        if ((not line_arr[2][0] == 'F') and ("\\" not in line_arr[1]) and ("," not in line_arr[1]) and
        ("/" not in line_arr[1]) and ("_" not in line_arr[1]) and ("www." not in line_arr[1]) and 
        (":" not in line_arr[1]) and (";" not in line_arr[1]) and ("." not in line_arr[1]) and 
        ("-" not in line_arr[1]) and (not line_arr[2] in ["NNP", "Z", "W","DT", "PDT", "CC", "WDT"])):
            words.append(line_arr[1].lower())

    #with open(output_path+"wiki2010_eng_{}".format(file_counter), 'wb') as output_datafile:
    #    pickle.dump(words, output_datafile)

    file_counter += 1
    all_words += words

del words
del lines

print("{} words found".format(len(all_words)))
unique_words = list(set(all_words))
print("{} unique words found".format(len(unique_words)))
print("Creating word indices...")
word_to_index = {word: index for index, word in enumerate(unique_words)}
numpy_corpus = np.array([word_to_index[w] for w in all_words])
#print(numpy_corpus[0:100])

print("Saving corpus...")
with open(output_path+"corpus", 'wb') as output_datafile:
    pickle.dump(numpy_corpus, output_datafile)

print("Saving dictionary...")
with open(output_path+"word_to_index", 'wb') as output_datafile2:
    pickle.dump(word_to_index, output_datafile2)

print("Done!")