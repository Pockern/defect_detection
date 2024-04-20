import logging
import argparse
import os
import numpy as np
import torch
import json
import random
import warnings

from model import Model
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (RobertaConfig, RobertaModel, RobertaForSequenceClassification, RobertaTokenizer, AdamW,
                          get_linear_schedule_with_warmup) 

# init logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

# class to shape training input features
class InputFeatures(object):
    def __init__(self, input_functions_tokens, input_functions_ids, functions_idx, functions_labels, file_label, file_idx):
        self.input_functions_tokens = input_functions_tokens
        self.input_functions_ids = input_functions_ids
        self.functions_idx = functions_idx
        self.functions_labels = functions_labels
        self.file_label = file_label
        self.file_idx = file_idx


def convert_examples_to_features(object, tokenizer, args, file_key):
    """
    convert obj of json file to features that model needs
    :return: inputfeatures
    """
    if file_key == 'functions_before_patches':
        file_type = 'bad'
    else:
        file_type = 'good'
    
    input_functions_tokens = []
    input_functions_ids = []
    functions_idx = []
    functions_labels = []
    file_idx = object['cwe'] + '/' + object['language'] + '/' + file_type + object['cwe_id']

    functions = object[file_key]
    if len(functions) == 0:
        return None
    else:
        file_label = functions[0]
        file_label = file_label['bag_label']
    for function in functions:
        code = ' '.join(function['function_code'].split())
        code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        input_functions_tokens.append(source_tokens)
        input_functions_ids.append(source_ids)
        functions_idx.append(function['function_idx'])
        functions_labels.append(function['function_label'])

    return InputFeatures(input_functions_tokens, input_functions_ids, functions_idx, functions_labels, file_label, file_idx)


# dataset class for get features from file with notification
class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                feature = convert_examples_to_features(js, tokenizer, args, 'functions_before_patches')
                if feature is not None:
                    self.examples.append(feature)
                feature = convert_examples_to_features(js, tokenizer, args, 'functions_after_patches')
                if feature is not None:
                    self.examples.append(feature)
                
                # self.examples.append(convert_examples_to_features(js, tokenizer, args, 'functions_before_patches'))
                # self.examples.append(convert_examples_to_features(js, tokenizer, args, 'functions_after_patches'))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:2]):
                    logger.info("*** Example ***")
                    logger.info("file_id: {}".format(example.file_idx))
                    logger.info('file label: {}'.format(example.file_label))
                    logger.info("number of functions: {}".format(len(example.functions_idx)))
                    logger.info("function labels: {}".format(example.functions_labels))
                    logger.info("first function tokens: {}".format([x.replace('\u0120','_') for x in example.input_functions_tokens[0]]))
                    logger.info("first function ids: {}".format(' '.join(map(str, example.input_functions_ids[0]))))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return torch.tensor(self.examples[index].input_functions_ids), torch.tensor(self.examples[index].functions_labels), torch.tensor(self.examples[index].file_label)


def set_seed(seed=42):
    """
    ensure reproducibility of results
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    # dataloader init
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size, num_workers=2, pin_memory=False)

    # parameter init
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader) // args.gradient_accumulation_steps
    args.warmup_stemps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    model.to(args.device)

    # prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # check if reload model is required
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))

    # begin to train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0.0
    model.zero_grad()

    for idx in range(args.start_epoch, int(args.epoch)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            functions_inputs = batch[0].to(args.device)
            functions_labels = batch[1].to(args.device)
            file_label = batch[2].to(args.device)

            model.train()
            loss, logits = model(functions_inputs, functions_labels, file_label)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} step {} loss {}".format(idx, step+1, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)),4)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # check whether eval during train
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, eval_during_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))

                    if results['eval_acc'] > best_acc:
                            best_acc = results['eval_acc']
                            logger.info("  "+"*"*20)  
                            logger.info("  Best acc:%s", round(best_acc, 4))
                            logger.info("  "+"*"*20)                          
                            
                            checkpoint_prefix = 'checkpoint-best-acc'
                            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)                        
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_dir = os.path.join(output_dir, '{}'.format(args.language_type + '_model.bin')) 
                            torch.save(model_to_save.state_dict(), output_dir)
                            logger.info("Saving model checkpoint to %s", output_dir) 


def evaluate(args, model, tokenizer, eval_during_training=False):
    # eval parameter init
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    if args.n_gpu > 1 and eval_during_training is False:
        model = torch.nn.DataParallel(model)

    # begin to eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]

    for batch in eval_dataloader:
        functions_ids = batch[0].to(args.device)
        functions_labels = batch[1].to(args.device)
        file_label = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(functions_ids, functions_labels, file_label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(file_label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits,0)
    labels = np.concatenate(labels,0)
    preds = logits[:,0] > 0.5
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc, 4),
    }
    return result


def test(args, model, tokenizer):
    # test parameter init
    test_dataset = TextDataset(tokenizer, args,args.test_data_file)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # begin to test
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    logits=[]   
    labels=[]

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        # TODO
        functions_ids = batch[0].to(args.device)
        functions_labels = batch[1].to(args.device)
        file_label = batch[2].to(args.device)
        with torch.no_grad():
            logit = model(functions_ids, functions_labels)
            logits.append(logit.cpu().numpy())
            labels.append(file_label.cpu().numpy())

    logits = np.concatenate(logits,0)
    labels = np.concatenate(labels,0)
    preds = logits[:,0] > 0.5
    with open(os.path.join(args.output_dir, args.language_type + "_predictions.txt"), 'w') as f:
        for example, pred in zip(test_dataset.examples, preds):
            if pred:
                f.write(str(example.file_idx)+'\t1\n')
            else:
                f.write(str(example.file_idx)+'\t0\n') 


def main():
    # parmeter setup ---------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--train_data_file", type=str, required=True, 
                        help="The input training data file (a json file)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--language_type", type=str, required=True, 
                        help="The PL of intput code")
    # other
    parser.add_argument("--eval_data_file", type=str,
                        help="The input evaluation data file (a json file)")
    parser.add_argument("--test_data_file", type=str,
                        help="The input test data file (a json file)")
    parser.add_argument("--model_name_or_path", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="whether to run eval.")
    parser.add_argument("--do_test", action='store_true',
                        help="whether to run test.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--epoch', type=int, default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')

    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------

    # setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    # setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # setup seed
    set_seed(args.seed)

    # check if reload epoch, step is required
    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1
        
        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())
        
        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    # load pretrained model, config and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    # model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path)

    model = Model(model, config, tokenizer, args)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-acc/' + args.language_type + '_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/' + args.language_type + '_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))                  
        model.to(args.device)
        test(args, model, tokenizer)

if __name__ == '__main__':
    main()



