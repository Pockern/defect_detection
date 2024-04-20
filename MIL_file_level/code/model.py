import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.dropout = nn.Dropout(args.dropout_probability)

    def forward(self, input_ids=None, labels=None):
        # 1 为 padding，需要mask忽略
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        outputs = self.dropout(outputs)

        # print('\ninputs:')
        # print(input_ids)
        # print(input_ids.shape)

        # print('\noutputs:')
        # print(outputs)
        # print(outputs.shape)

        # print('\nlabels:')
        # print(labels)
        # print(labels.shape)

        logits = outputs
        prob = torch.sigmoid(logits)
 
        # print('\nprob[:, 0]')
        # t1 = prob[:,0] + 1e-10
        # print(t1)
        # print(t1.shape)

        # print('\n**:')
        # answer = t1*labels
        # print(answer.shape)
        # print(torch.log(answer))

        # print('\nloss:')        
        # loss = torch.log(prob[:,0]+1e-10)*labels + torch.log((1-prob)[:,0]+1e-10)*(1-labels)
        # print(loss)
        # print(loss.shape)

        # print('\nloss.mean:')
        # loss = -loss.mean()
        # print(loss)
        # print(loss.shape)

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:,0]+1e-10)*labels + torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob