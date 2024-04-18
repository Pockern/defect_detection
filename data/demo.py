import json

with open('cpp_divided.jsonl') as f:
    lines = f.readlines()

js = json.loads(lines[126])
functions_before_label = [fuc['function_label'] for fuc in js['functions_before_patches']]
print(functions_before_label)
print(js['cwe'] + js['cwe_id'])