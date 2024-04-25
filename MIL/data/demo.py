import json
import random
import os

def len_of_json(file_name):
    js_objects = []
    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            js_objects.append(data)

    print('{} len: {}'.format(file_name, len(js_objects)))

def function_state(file_name):
    js_objects = []
    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            js_objects.append(data)

    len_state = {}
    for object in js_objects:
        len1 = len(object['functions'][0])
        if len1 in len_state:
            len_state[len1] += 1
        else:
            len_state[len1] = 1

    print(file_name)
    for key in sorted(len_state.keys()):
        count = len_state[key]
        print("len {} : {}".format(key, count))


class DevignDataEntry:
    def __init__(self, idx, func, target):
        self.idx = idx
        self.func = func
        self.target = target

    def to_dict(self):
        return {
            'idx': self.idx, 
            'func': self.func,
            'target': self.target
        }


def change_to_dvign(file_name):
    dir = 'testdata'
    if not os.path.exists(dir):
        os.makedirs(dir)
    if 'train' in file_name:
        output_dir = os.path.join(dir, 'train.jsonl')
    elif 'valid' in file_name:
        output_dir = os.path.join(dir, 'valid.jsonl')
    else:
        output_dir = os.path.join(dir, 'test.jsonl')

    objects = []
    with open(file_name, 'r') as f:
        for line in f:
            objects.append(json.loads(line))

    outputs = []
    for object in objects:
        functions_before_patches = object['functions_before_patches']
        if len(functions_before_patches) > 0:
            for function in functions_before_patches:
                idx = function['function_idx']
                func = function['function_code']
                target = function['function_label']
                outputs.append(DevignDataEntry(idx, func, target).to_dict())

        functions_after_patches = object['functions_after_patches']
        if len(functions_after_patches) > 0:
            for function in functions_after_patches:
                idx = function['function_idx']
                func = function['function_code']
                target = function['function_label']
                outputs.append(DevignDataEntry(idx, func, target).to_dict())

    print(output_dir) 
    tag1 = [object for object in outputs if object['target'] == 1]
    tag0 = [object for object in outputs if object['target'] == 0]
    print(len(tag1))
    print(len(tag0))

    outputs1 = []
    for data in tag1:
        outputs1.append(data)
    for data in tag0[:1000]:
        outputs1.append(data)

    with open(output_dir, 'w') as f:
        for data in outputs1:
            json.dump(data, f)
            f.write('\n')

    tag1 = [object for object in outputs1 if object['target'] == 1]
    tag0 = [object for object in outputs1 if object['target'] == 0]
    print(len(tag1))
    print(len(tag0))
    

def bag_state(file_name):
    with open(file_name, 'r') as f:
        objects = [json.loads(line) for line in f]

    pos = 0
    neg = 0
    for object in objects:
        if object['file_label'] == 1:
            neg += 1
        else:
            pos += 1
    print('{}: good bag {} / bad bag {}'.format(file_name, pos, neg))


def has_repeat(file_name):
    file_list = []
    tag = 0
    with open(file_name, 'r') as f:
        objects = [json.loads(line) for line in f]
    
    for object in objects:
        if not object['file_idx'] in file_list:
            file_list.append(object['file_idx'])
        else:
            tag = 1
            print('{}'.format(object['file_idx']))

    if tag == 0:
        print('no repeat file')


def main():
    for language in ['c', 'cpp', 'py', 'java', 'js']:
        print('\n')
        for file in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
            file_name = os.path.join(language, file)

            # function_state(file_name)
            len_of_json(file_name)
            # bag_state(file_name)
            # has_repeat(file_name)
        


if __name__ == '__main__':
    main()