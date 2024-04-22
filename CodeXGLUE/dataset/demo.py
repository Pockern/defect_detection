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

def len_state(file_name):
    js_objects = []
    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            js_objects.append(data)

    len_state = {}
    for object in js_objects:
        len1 = len(object['functions_before_patches'])
        if len1 in len_state:
            len_state[len1] += 1
        else:
            len_state[len1] = 1

        len1 = len(object['functions_after_patches'])
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
    


def main():
    for file_name in ['./testdata/train.jsonl', './testdata/valid.jsonl', './testdata/test.jsonl']:
        # len_state(file_name)
        len_of_json(file_name)
    
    # file_name = './c/test.jsonl'
    # change_to_dvign(file_name)
        


if __name__ == '__main__':
    main()