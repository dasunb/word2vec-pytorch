import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader
import numpy as np
from statistics import mean
import os
from model import Word2VecNet, Word2VecDataset
import math
import pickle
import pandas as pd
import gc



def predict(test_dataloader, model, batch_size, embedding_dim, learning_rate):
    epochs = 10
    last_epoch = 500
    version = "v9"
    #loss_row = []
    while (epochs <= last_epoch):
       
        MODEL_PATH = "saved_models/batch_size_{}_dim_{}_lr_{}_ER_False_Epochs_{}_{}.pt".format(batch_size, embedding_dim, learning_rate, epochs, version)
        
        print("loading model checkpoint: " + MODEL_PATH)

        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        test_losses = []
        
        for _, (x, y, neg) in enumerate(test_dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)

            with torch.no_grad():
                test_loss = model.predict(x, y, neg)
                test_losses.append(test_loss.detach().item())
        
        #loss_row.append([epochs, mean(test_losses)])

        print({'epoch': epochs, 'loss': mean(test_losses)})

        save_path_loss =  "results/result_batch_size_{}_dim_{}_lr_{}_ER_False_{}_test.csv".format(batch_size, embedding_dim, learning_rate, version)

        df_results = pd.DataFrame([[epochs, mean(test_losses)]], columns=["Epoch", "test_loss"])

        if not os.path.isfile(save_path_loss):
            df_results.to_csv(save_path_loss, sep='\t', header='column_names')
        else:
            df_results.to_csv(save_path_loss, sep='\t', mode='a', header=False)
  
        #loss_row = []

        epochs += 10


if __name__ == '__main__':
    max_epochs = 500 
    batch_size = 20000
    embedding_dim = 200 
    learning_rate = 0.001 
    default_context_size = 4
    split_val = 0.8
    num_negative = 5

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU available and utilized")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    #Dataset path
    path = "data/"

    results_path = 'results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    #dataset = load_dataset("data/", 4, 8, batch_size, max_epochs)

    corpus = []
    seq_len = default_context_size

    print("Loading corpus...")
    with (open(path+"filtered_corpus", 'rb')) as filehandle_1:
        corpus = pickle.load(filehandle_1)

    corpus = corpus.astype(np.int32)
    #corpus = np.array(corpus, dtype=np.int32)
    print("Total corpus size: {}".format(corpus.shape[0]))
    test_corpus = corpus[math.floor(split_val*corpus.shape[0]):]
    #FOR TESTING
    #test_corpus = test_corpus[:100000]

    unique_words = list(set(corpus))
    num_samples, num_unique_words = test_corpus.shape[0], len(unique_words)
    data_len = num_samples - seq_len

    del corpus
    gc.collect()

    #num_negative_samples = data_len

    print("Loading x values...")
    with (open(path+"x_test", 'rb')) as filehandle_2:
        shared_array_x = pickle.load(filehandle_2)

    print("Loading y values...")
    with (open(path+"y_test", 'rb')) as filehandle_3:
        shared_array_y = pickle.load(filehandle_3)

    print("Loading Unigram table...")
    with (open(path+"unigram_table_3", 'rb')) as filehandle_4:
        unigram_table = pickle.load(filehandle_4)

    print("Shape of x values: {}".format(shared_array_x.shape))
    print("Shape of y values: {}".format(shared_array_y.shape))
    print("Shape of unigram table: {}".format(unigram_table.shape))

    print("Creating dataset...")
    test_dataset = Word2VecDataset(shared_array_x, shared_array_y, unigram_table, num_unique_words, num_negative, len(unigram_table))

    print("Datasets created")
    print("training corpus size {}".format(len(test_dataset)))
    print("unique words {}".format(test_dataset.num_unique()))
    model = Word2VecNet(seq_len, test_dataset.num_unique(), embedding_dim).to(device)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=True, pin_memory=True,
                            num_workers=16, prefetch_factor=2, persistent_workers=True)
    predict(test_dataloader, model, batch_size, embedding_dim, learning_rate)