import torch
import torch.nn as nn
from torch.nn import functional as from
torch.manual_seed(1708) #for reproducibility

class BigramLanguageModel(nn.Module):
    def __init__(self,
                vocabulary_size:int,
                n_embedding_dimension:int,
                block_size:int
                ):
        #define the base variables
        self.token_embedding_table = nn.Embedding(vocabulary_size,n_embedding_dimension)
        self.position_embedding_table = nn.Embedding(block_size,n_embedding_dimension)
        #this is gonna be useful to map the embedding dimension to the target vocabulary
        self.lm_head =  nn.Linear(n_embedding_dimension,vocabulary_size)

    def forward(self,idx,targets=None):
        idx = idx.to(device)
        #extract the variables related to the input
        B,T = idx.shape
        #time, to obtain the values for the input sequence
        token_embeddings = self.token_embedding_table(idx)
        #then, we are gonna obtain the positional embeddings
        indices = torch.arange(T,device=device)
        position_embeddings = self.position_embedding_table(indices % self.position_embedding_table.weight.size(0))
        #it's time to create the input
        x = token_embeddings + position_embeddings
        #apply the linear layer to the input
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape #C is not the embedding size, but is the vocabulary size
            logits = logits.view(B*T,C)
            target = targets.view(B*T)
            loss = F.cross_entropy(logits,target)
            return logits, loss
    def generate(self,idx,max_new_tokens):
        #time to generate the new tokens
        for _  in range(max_new_tokens):
            logits,_ = self(idx)
            #keep the logits of the last token
            logits  = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            #concat with the past samples
             idx = torch.cat((idx,idx_next),dim=1)
        return idx
