import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # self.dropout = nn.Dropout(args.dropout_probability)
        self.classifier = nn.Sequential(
            nn.Linear(768, 2),
            nn.Sigmoid()
        )

    def forward(self, input_ids=None, labels=None):
        if self.args.model_name == 'codet5-base':
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs['decoder_hidden_states'][-1]
            eos_mask = input_ids.eq(self.config.eos_token_id)

            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        else:
            # 1 为 padding，需要mask忽略
            outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
            outputs = outputs.pooler_output

        # outputs = self.dropout(outputs)

        logits = self.classifier(outputs)
        prob = logits

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:,0]+1e-10)*labels + torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
    
    def cal_precision(self, labels, preds):
        return precision_score(labels, preds, zero_division=0)

    def cal_recall(self, labels, preds):
        return recall_score(labels, preds, zero_division=0)
    
    def cal_f1(self, labels, preds):
        return f1_score(labels, preds, zero_division=0)
    
    def cal_acc(self, labels, preds):
        return accuracy_score(labels, preds)