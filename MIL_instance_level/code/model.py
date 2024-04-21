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
        self.linear = nn.Linear(in_features=768, out_features=2)

    def forward(self, input_functions_ids=None, functions_labels=None, file_label=None):
        # 1 为 padding，需要mask忽略

        # print('\ninput_functions:')
        # print(input_functions_ids)
        # print(input_functions_ids.shape)
        # print("\nfunctions labels:")
        # print(functions_labels)
        # print(functions_labels.shape)
        # print('\nfile_labels:')
        # print(file_label)
        # print(file_label.shape)
        # print('\ninput_functions_ids[0]: ')
        # print(input_functions_ids[0])
        # print(input_functions_ids[0].shape)
        # print('\noutputs')
        # outputs = self.encoder(input_functions_ids[0], attention_mask=input_functions_ids[0].ne(1))
        # print(outputs)
        # print('\npooler_outputs:')
        # outputs_pooler = outputs.pooler_output
        # print(outputs_pooler)
        # print(outputs_pooler.shape)
        # exit(0)

        outputs = self.encoder(input_functions_ids[0], attention_mask=input_functions_ids[0].ne(1))
        outputs = outputs.pooler_output
    
        outputs = self.dropout(outputs)
        bag_feature = nn.functional.adaptive_max_pool2d(outputs.unsqueeze(0), (1, 768))[0]

        logits = self.linear(bag_feature)
        prob = torch.sigmoid(logits)

        if file_label is not None:
            file_label = file_label.float()
            loss = torch.log(prob[:,0]+1e-10)*file_label + torch.log((1-prob)[:,0]+1e-10)*(1-file_label)
            loss = -loss
            return loss, prob
        else:
            return prob