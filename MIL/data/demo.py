import json

def main():
    file_name = './cpp_divided.jsonl'
    js_objects = []
    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            js_objects.append(data)

    ids = [object['file_idx'] for object in js_objects]
    print(ids)
    


if __name__ == '__main__':
    main()