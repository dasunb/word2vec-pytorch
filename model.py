import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import ctypes
#import multiprocessing as mp
from tqdm import tqdm
import random
from random import randint
#from numpy.random import randint
import gc

#Dataset Class
class Word2VecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x,
        y,
        unigram_table,
        num_unique_words,
        num_negative,
        unigram_len

    ):
        self.num_unique_words = num_unique_words
        self.num_negative = num_negative

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        
        self.unigram_table = torch.from_numpy(unigram_table)
        self.unigram_len = unigram_len
        self.__data_len = self.y.shape[0]

    def num_unique(self):
        return self.num_unique_words

    def __len__(self):
        return self.__data_len

    def __getitem__(self, index):
        
        return (      
           self.x[index],
           self.y[index],
           #Get random samples from unigram table for each training item
           self.unigram_table[torch.randint(self.unigram_len, (self.num_negative,))]
        )

#CBOW model
class Word2VecNet(nn.Module):
    def __init__(self, seq_len, vocab_size, embedding_size):
        super(Word2VecNet, self).__init__()
        self.context_size = seq_len
        self.embedding_dim = embedding_size
        self.vocab_size = vocab_size

        #Embedding layer to hold input "context" embeddings
        self.embeddings_in = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            sparse=True,
        )

        #Embedding layer to hold output "target" embeddings
        self.embeddings_out = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            sparse=True,
        )

        initrange = 1.0 / self.embedding_dim
        #initialize the input embeddings from a uniform distribution
        init.uniform_(self.embeddings_in.weight.data, -initrange, initrange)
        #initialize the output embeddings to zero
        init.constant_(self.embeddings_out.weight.data, 0)

    def forward(self, inputs, target_word, negative_samples):
        #Embedding of the target word (vocab_size x embedding_dim)
        embeds_out = self.embeddings_out(target_word)
 
        #Embeddings of the input words (context size x vocab_size x embedding_dim)
        embeds_in = self.embeddings_in(inputs)
   
        #Sum up the input context words to get the "embedding bag" (vocab_size x embedding_dim)
        embeds_in_bag = torch.sum(embeds_in, dim=1)
   
        #Multiply the input embeddings with the target embeddings and apply logsigmoid activation function and sum along dim 1 to get the 'positive score'
        pos_out = torch.sum(F.logsigmoid(torch.clamp(torch.mul(embeds_in_bag, embeds_out), max=10, min=-10)), dim=1)

        neg_embeddings = self.embeddings_out(negative_samples)

        #batch matrix multiply the negative samlpe embeddings with the output embeddings
        embeds_product_neg = torch.bmm(neg_embeddings, embeds_out.unsqueeze(2)).squeeze()

        #apply logsigmoid activation function and sum along dim 1 to get the 'negative score'
        neg_score =  torch.sum(F.logsigmoid(-(torch.clamp(embeds_product_neg, max=10, min=-10))), dim=1)

        #Sum up the positive and negative scores and get its mean to get the loss
        return -torch.mean(pos_out + neg_score)

    def get_embedding(self, input_word):
        return(self.embeddings_in(input_word))

    def get_random_embeddings(self, inputs, sample_size, rng):
        input_indices = inputs[torch.tensor(rng.choice(np.arange(len(inputs)), sample_size, replace=False))]
        return(self.embeddings_in(input_indices))


    def predict(self, inputs, target_word, negative_samples):
        res_prob = self.forward(inputs, target_word, negative_samples)
        return res_prob