import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.M = 768
        self.L = 128
        self.feature_channel = 3
        # self.feature_channel = 1
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.feature_channel)
        )

        self.conv_l1 = torch.nn.Conv1d(self.feature_channel, self.feature_channel, 3, stride=3)
        self.conv_l2 = torch.nn.Conv1d(self.feature_channel, self.feature_channel, 1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M, 2),
            nn.Sigmoid()
        )

    def forward(self, input_functions_ids=None, functions_labels=None, file_label=None):
        if self.args.model_name == 'codet5-base':
            input_functions_ids = input_functions_ids.view(-1, self.args.block_size)
            attention_mask = input_functions_ids.ne(self.tokenizer.pad_token_id)
            outputs = self.encoder(input_ids=input_functions_ids, attention_mask=attention_mask,
                               labels=input_functions_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs['decoder_hidden_states'][-1]
            eos_mask = input_functions_ids.eq(self.config.eos_token_id)

            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]

        else:
            outputs = self.encoder(input_functions_ids[0], attention_mask=input_functions_ids[0].ne(1))
            # [k, 768] k个function
            outputs = outputs.pooler_output

        # [k, 3] functions权重
        A = self.attention(outputs)  # KxATTENTION_BRANCHES
        # [3, k] 转置+归一
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        # key[3, k] * value[k, 768]
        # [3, 768] 加权和
        Z = torch.mm(A, outputs)  # ATTENTION_BRANCHESxM

        # [3, 256]
        Z1 = F.relu(self.conv_l1(Z))
        # [3, 256]
        Z2 = F.relu(self.conv_l2(Z1))
        # [1, 768]
        Z3 = Z2.view(1, -1)

        # [1, 2] tag
        # prob_file = self.classifier(Z)
        prob_file = self.classifier(Z3)
        # [k, 2] tag
        prob_functions = self.classifier(outputs)

        if file_label is not None:
            file_label = file_label.float()
            functions_labels = functions_labels.float() 

            # loss=torch.log(prob_file[:,0]+1e-10)*file_label+torch.log((1-prob_file)[:,0]+1e-10)*(1-file_label)
            # loss = -loss.mean()
            
            # 根据消融对应修改 args.output_dir
            loss_file = torch.log(prob_file[:,0]+1e-10)*file_label + torch.log((1-prob_file)[:,0]+1e-10)*(1-file_label)
            loss_functions = torch.log(prob_functions[:,0]+1e-10)*functions_labels + torch.log((1-prob_functions)[:,0]+1e-10)*(1-functions_labels)
            loss = -(0.5 * loss_file + 0.5 * loss_functions.mean())
            
            return loss, prob_file, A
        else:
            return prob_file, A
        
    def cal_precision(self, labels, preds):
        return precision_score(labels, preds, zero_division=0)

    def cal_recall(self, labels, preds):
        return recall_score(labels, preds, zero_division=0)
    
    def cal_f1(self, labels, preds):
        return f1_score(labels, preds, zero_division=0)
    
    def cal_acc(self, labels, preds):
        return accuracy_score(labels, preds)