import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import torch.nn.functional as F
import copy

class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = config
        self.classifier = RobertaClassificationHead(config)
        self.args=args
    
    def forward(self, input_ids=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        outputs = self.encoder(input_ids = input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs) 
        probability = F.softmax(logits) #To obtain the probabilities for each label
        if labels is not None:
            loss_function = CrossEntropyLoss()
            loss = loss_function(logits, labels)
            return loss, probability
        else:
            return probability
      
        
 
        


