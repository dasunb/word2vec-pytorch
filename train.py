import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from statistics import mean
import os
from model import Word2VecNet, Word2VecDataset
import pickle
import pandas as pd
import time
import gc
import argparse

def train(model, train_dataloader, optimizer, batch_size, embedding_dim, device, last_epoch, version):
    model.train()
        
    training_losses = []
    for epoch in range(1 + last_epoch, max_epochs + 1):
        start = time.time()
        losses = []
        for _, (x, y, neg) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)

            loss = model(x, y, neg)
            loss.backward()
            losses.append(loss.detach().item())

            optimizer.step()

        end = time.time()
        print({'epoch': epoch, 'loss': mean(losses), 'time elapsed': end-start})
        training_losses.append([epoch, mean(losses)])

        if(epoch % 10 == 0):           
            SAVE_MODEL_PATH = "saved_models/batch_size_{}_dim_{}_lr_{}_ER_False_Epochs_{}_{}.pt".format(batch_size, embedding_dim, learning_rate, epoch, version)
            SAVE_SUMMARY_LOSS_PATH =  "results/result_batch_size_{}_dim_{}_lr_{}_ER_False_{}.csv".format(batch_size, embedding_dim, learning_rate, version)
            SAVE_LOSS_PATH = "results/result_batch_size_{}_dim_{}_lr_{}_ER_False_{}_full.csv".format(batch_size, embedding_dim, learning_rate, version)

            #Save checkpoint for later reference
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, SAVE_MODEL_PATH)

            #Save losses and loss summary
            df_results_summary = pd.DataFrame([[epoch, mean(losses)]], columns=["Epoch", "train_loss"])
            df_results_full = pd.DataFrame(training_losses, columns=["Epoch", "train_loss"])

            if not os.path.isfile(SAVE_SUMMARY_LOSS_PATH):
                df_results_summary.to_csv(SAVE_SUMMARY_LOSS_PATH, sep='\t', header='column_names')               
            else:
                df_results_summary.to_csv(SAVE_SUMMARY_LOSS_PATH, sep='\t', mode='a', header=False)
                

            if not os.path.isfile(SAVE_LOSS_PATH):
                df_results_full.to_csv(SAVE_LOSS_PATH, sep='\t', header='column_names')
            else:
                df_results_full.to_csv(SAVE_LOSS_PATH, sep='\t', mode='a', header=False)

            training_losses = []


if __name__ == '__main__':
    #Default Training variables
    default_context_size = 4
    split_val = 0.8
    num_negative = 5
    
    #Pass Training variables as arguements
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--embedding_dim", dest="embedding_dim", type=int, default=200, help="Embedding Dimmension Size")
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=20000,  help="Batch Size")
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-m", "--max_epochs", dest="max_epochs", type=float, default=500, help="Stop training at epoch")
    parser.add_argument("-c", "--last_epoch", type=int, dest="last_epoch", default=0, help="resume from checkpoint created at epoch")
    parser.add_argument("-v", "--version", type=int, dest="version", default=0, help="experiment version")

    args = parser.parse_args()

    max_epochs = args.max_epochs 
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    learning_rate = args.learning_rate 
    last_epoch = args.last_epoch
    version = args.version

    print("Training variable summary")
    print("Embedding dimensions: {}".format(embedding_dim))
    print("Batch size: {}".format(batch_size))
    print("Learning rate: {}".format(learning_rate))
    print("Input Context size: {}".format(default_context_size))
    print("Negative samples per training item: {}".format(num_negative))


    #Utilize Cuda if available
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
    
    #Load corpus
    corpus = []
    seq_len = default_context_size

    print("Loading corpus...")
    with (open(path+"filtered_corpus", 'rb')) as filehandle_1:
        corpus = pickle.load(filehandle_1)

    corpus = corpus.astype(np.int32)

    #FOR TESTING
    #train_corpus = corpus[:10000000]

    unique_words = list(set(corpus))
    del corpus
    gc.collect()

   
    #Load cached training items and unigram tables to avoid unnecessary data preprocessing each time
    print("Loading x values...")
    with (open(path+"x_train", 'rb')) as filehandle_2:
        shared_array_x = pickle.load(filehandle_2)

    print("Loading y values...")
    with (open(path+"y_train", 'rb')) as filehandle_3:
        shared_array_y = pickle.load(filehandle_3)

    print("Loading Unigram table...")
    with (open(path+"unigram_table", 'rb')) as filehandle_4:
        unigram_table = pickle.load(filehandle_4)

    unigram_table = unigram_table.astype(np.int32)
    num_samples, num_unique_words = shared_array_y.shape[0], len(unique_words)
    data_len = num_samples - seq_len


    print("Shape of x items: {}".format(shared_array_x.shape))
    print("Shape of y items: {}".format(shared_array_y.shape))
    print("Shape of unigram table: {}".format(unigram_table.shape))
    print("Creating dataset...")
    train_dataset = Word2VecDataset(shared_array_x, shared_array_y, unigram_table, num_unique_words, num_negative, len(unigram_table))

    print("Datasets created")
    print("training dataset size: {}".format(len(train_dataset)))
    print("number of unique words in corpus: {}".format(train_dataset.num_unique()))

    model = Word2VecNet(seq_len, train_dataset.num_unique(), embedding_dim).to(device)
    optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate)

    if last_epoch > 0:
        MODEL_PATH = "saved_models/batch_size_{}_dim_{}_lr_{}_ER_False_Epochs_{}_{}.pt".format(batch_size, embedding_dim, learning_rate, last_epoch, version)
        
        print("loading last model checkpoint: " + MODEL_PATH)

        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']   
    

    model.share_memory()

    #Define dataloader with multiprocessing enabled by setting num_workers > 1 to ensure efficient dataloading if training on GPU
    num_processors = 1
    if is_cuda == True:
        num_processors = 16

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, drop_last=True, pin_memory=True, 
                            num_workers=num_processors, prefetch_factor=2, persistent_workers=True)
                            
    print("Training...")
    train(model, train_dataloader, optimizer, batch_size, embedding_dim, device, last_epoch, version)
