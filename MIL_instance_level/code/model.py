import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.M = 768
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 2),
            nn.Sigmoid()
        )

    def forward(self, input_functions_ids=None, functions_labels=None, file_label=None):

        outputs = self.encoder(input_functions_ids[0], attention_mask=input_functions_ids[0].ne(1))
        # [k, 768] k个function
        outputs = outputs.pooler_output

        # [k, 1] 注意力权重
        A = self.attention(outputs)  # KxATTENTION_BRANCHES
        # [1, 9] 转置+归一
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        # key[1, k] * value[k, 768]
        # [1, 768] 加权和
        Z = torch.mm(A, outputs)  # ATTENTION_BRANCHESxM

        # [1, 2] tag
        prob_file = self.classifier(Z)
        # [k, 2] tag
        # prob_functions = self.classifier(outputs)

        if file_label is not None:
            file_label = file_label.float()
            functions_labels = functions_labels.float()
            loss_file = torch.log(prob_file[:,0]+1e-10)*file_label + torch.log((1-prob_file)[:,0]+1e-10)*(1-file_label)
            # loss_functions = torch.log(prob_functions[:,0]+1e-10)*functions_labels + torch.log((1-prob_functions)[:,0]+1e-10)*(1-functions_labels)
            # loss = -(0.5 * loss_file + 0.5 * loss_functions.mean())
            loss=torch.log(prob_file[:,0]+1e-10)*file_label+torch.log((1-prob_file)[:,0]+1e-10)*(1-file_label)
            loss = -loss.mean()
            return loss, prob_file, A
        else:
            return prob_file, A


    def cal_auc_score(self, file_labels, probs):
        auc = roc_auc_score(file_labels, probs)
        return auc