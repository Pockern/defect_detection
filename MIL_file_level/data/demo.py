import json

def main():
    file_name = './py/train.jsonl'
    with open(file_name, 'r') as f:
        data = [json.loads(item) for item in f]

    # ids = [object['file_idx'] for object in data]
    print(data[0][ 'before'].keys())
    print(data[0].keys())
    js = data[0]
    print(js['cwe']+'/'+js['language']+'/bad'+js['cwe_id'])
    print(js['language'])
    


if __name__ == '__main__':
    main()