import json
import random

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


def main():
    for file_name in ['./c/train.jsonl', './cpp/train.jsonl', './py/train.jsonl']:
        # len_state(file_name)
        len_of_json(file_name)


if __name__ == '__main__':
    main()