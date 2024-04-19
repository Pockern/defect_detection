import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # self.dropout = nn.Dropout(args.dropout_probability)

    def forward(self, input_functions_ids=None, functions_labels=None, file_label=None):
        # 1 为 padding，需要mask忽略

        print(input_functions_ids)
        print(input_functions_ids.shape)
        print(functions_labels)
        print(functions_labels.shape)
        print(file_label)
        outputs = self.encoder(input_functions_ids[0], attention_mask=input_functions_ids[0].ne(1))
        print(input_functions_ids[0].shape)
        print(dir(outputs))
        outputs_last = outputs.last_hidden_state
        outputs_pooler = outputs.pooler_output
        print(outputs_last)
        print(outputs_last.shape)
        print(outputs_pooler)
        print(outputs_pooler.shape)
        exit(0)

        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        outputs = self.dropout(outputs)

        logits = outputs
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:,0]+1e-10)*labels + torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob