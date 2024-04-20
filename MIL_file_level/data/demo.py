import json

def main():
    file_name = './cpp_divided.jsonl'
    js_objects = []
    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            js_objects.append(data)

    # ids = [object['file_idx'] for object in js_objects]
    # js = js_objects[0]
    # print(js.keys())
    # functions_before = js['functions_before_patches']
    # print(functions_before[0].keys())
    
    output_dir = './temp.jsonl'
    cnt = 0
    dest_js = []
    for object in js_objects:
        functions = object['functions_before_patches']
        if len(functions) > 1:
            dest_js.append(object)
            cnt += 1
        if cnt == 32:
            break
    
    with open(output_dir, 'w') as f:
        for data in dest_js:
            json.dump(data, f)
            f.write('\n')



if __name__ == '__main__':
    main()