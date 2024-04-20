import json
import random

def main():
    # file_name = './cpp_divided.jsonl'
    file_name = './temp.jsonl'
    js_objects = []
    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            js_objects.append(data)


    print(len(js_objects))
    # output_dir = './temp.jsonl'
    # cnt = 0
    # dest_js = []
    # random.shuffle(js_objects)
    # for object in js_objects:
    #     dest_js.append(object)
    #     cnt += 1
    #     if cnt == 32:
    #         print('ok')
    #         break

    
    # print('from {} collect {}'.format(len(js_objects), len(dest_js)))

    # with open(output_dir, 'w') as f:
    #     for data in dest_js:
    #         json.dump(data, f)
    #         f.write('\n')

    object_len = [len(object['functions_before_patches']) for object in js_objects]
    print(max(object_len))
    # print(object_len)


if __name__ == '__main__':
    main()