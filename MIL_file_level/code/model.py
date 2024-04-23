import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.dropout = nn.Dropout(args.dropout_probability)
        self.classifier = nn.Sequential(
            nn.Linear(768, 2),
            nn.Sigmoid()
        )

    def forward(self, input_ids=None, labels=None):
        # 1 为 padding，需要mask忽略
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        outputs = outputs.pooler_output

        outputs = self.dropout(outputs)

        logits = self.classifier(outputs)
        prob = logits

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:,0]+1e-10)*labels + torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
    
    def cal_auc_score(self, file_labels, probs):
        auc = roc_auc_score(file_labels, probs)
        return auc