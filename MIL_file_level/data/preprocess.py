import json
import os
import re
import subprocess
import warnings
import random

from tree_sitter import Language, Parser

warnings.filterwarnings('ignore', category=FutureWarning)

# # 构筑语言模块
# Language.build_library(
#     'parser/languages.so',
#     [
#         'vendor/tree-sitter-c',
#         'vendor/tree-sitter-cpp',
#         'vendor/tree-sitter-python',
#         'vendor/tree-sitter-javascript'
#     ]
# )


class FileEntry:
    def __init__(self, file_idx, before, after, patches, language, cwe, cwe_id):
        self.file_idx = file_idx
        self.before = before
        self.after = after
        self.patches = patches
        self.language = language
        self.cwe = cwe
        self.cwe_id = cwe_id
    
    def to_dict(self):
        return {
            "file_idx": self.file_idx, 
            "before": self.before,              # ['file_code', 'file_label']
            "after": self.after,                # ['file_code', 'file_label']
            "patches": self.patches,
            "language": self.language,
            "cwe": self.cwe,
            "cwe_id": self.cwe_id
            # "functions_before_patches"        # list of FunctionEntry
            # "functions_after_patches"
        }
    

class DatasetEntry:
    def __init__(self, file_idx, file_code, language, file_label):
        self.file_idx = file_idx
        self.file_code = file_code,
        self.language = language,
        self.file_label = file_label

    def to_dict(self):
        return {
            "file_idx": self.file_idx,
            "file_code": self.file_code,
            "language": self.language,
            "file_label": self.file_label
        }


def remove_comment(code, language):
    """
    remove all none-code part like comment from code
    :param code: code with comments and so on
    :param language: language of code
    :return: cleaned code
    """
    if language == 'c' or language == 'cpp':
        pattern = r'//.*?$|/\*.*?\*/|/\*[\s\S]*?\*/|\*.*?$'
    elif language == 'py':
        pattern = r'#.*?$|#$|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|""".*?"""'
        language = 'python'

    # 加载语言模块
    LANGUAGE = Language('parser/languages.so', language)

    parser = Parser()
    parser.set_language(LANGUAGE)
    code = code.encode()
    tree = parser.parse(code)
    root_node = tree.root_node

    # 层次遍历
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type == 'comment':
            code = code[:node.start_byte] + code[node.end_byte:]
        else: 
            stack.extend(node.children)

    result =  code.decode()

    cleaned_code = re.sub(pattern, '', result, flags=re.MULTILINE)

    return cleaned_code


def dump_files_by_language_from_subfolder(root_folder, language, output_dir):
    """
    convert all files of language from root_folder into json
    :param root_folder: destination folder
    :param language: destination language
    """
    js = []
    file_idx = 0
    pattern = r"CWE-\d+"
    for root, dirs, files in os.walk(root_folder):
        if language in dirs:
            sub_foler_path = os.path.join(root, language)
            for sub_root, sub_dirs, sub_files in os.walk(sub_foler_path):
                cwe = re.findall(pattern, sub_root)
                for file in sub_files:
                    if 'bad' in file:
                        cwe_id = file.replace('bad', '')

                        # process file without patches
                        # 不能在这里直接消除代码中的注释，否则会和diff指令产生的行号出现偏差（diff的文件是原文件）
                        before = {}
                        file_before_path = os.path.join(sub_root, file)
                        with open(file_before_path, 'r') as f:
                            content = f.read()
                            before['file_code'] = content
                            before['file_label'] = 1

                        # preprocess file with patches
                        after = {}
                        file_after_path = file_before_path.replace('bad', 'good')
                        with open(file_after_path, 'r') as f:
                            content = f.read()
                            after['file_code'] = content
                            after['file_label'] = 0

                        # get patches
                        results = subprocess.run(["diff", "-u", file_before_path, file_after_path], capture_output=True, text=True)
                        patches = results.stdout

                        if len(cwe) > 0:
                            js.append(FileEntry(file_idx, before, after, patches, language, cwe[0], cwe_id).to_dict())
                        else:
                            js.append(FileEntry(file_idx, before, after, patches, language, 'None', cwe_id).to_dict())
                        file_idx += 1

    with open(output_dir, 'w') as f:
        for data in js:
            json.dump(data, f)
            f.write('\n')
    print('collect {} pairs of good/bad files from {}'.format(file_idx+1, language))


def split_dataset(file_name, language):
    output_dir = language
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # parameter init
    train_data_file = os.path.join(output_dir, 'train.jsonl')
    valid_data_file = os.path.join(output_dir, 'valid.jsonl')
    test_data_file = os.path.join(output_dir, 'test.jsonl')
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    with open(file_name, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    files = []
    for cnt, object in enumerate(data):
        if cnt % 2 == 0:
            file_idx = object['cwe'] + '/bad'+object['cwe_id']
            file_code = remove_comment(object['before']['file_code'], language)
            file_code = re.sub(r"[\n\t]", "", file_code)
            file_label = 1
            files.append(DatasetEntry(file_idx, file_code, language, file_label).to_dict())
        else:
            file_idx = object['cwe'] + '/good'+object['cwe_id']
            file_code = remove_comment(object['after']['file_code'], language)
            file_code = re.sub(r"[\n\t]", "", file_code)
            file_label = 0
            files.append(DatasetEntry(file_idx, file_code, language, file_label).to_dict())

    
    total_len = len(files)
    random.shuffle(files)
    train_len = int(total_len * train_ratio)
    valid_len = int(total_len * valid_ratio)
    test_len = total_len - train_len - valid_len

    train_data = files[:train_len]
    valid_data = files[train_len:train_len+valid_len]
    test_data = files[train_len+valid_len:]

    with open(train_data_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            json.dump(item, f)
            f.write('\n')

    with open(valid_data_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            json.dump(item, f)
            f.write('\n')
    
    with open(test_data_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            json.dump(item, f)
            f.write('\n')

    print('{} -- train dataset: {}, valid dataset: {}, test dataset: {}'.format(language, train_len, valid_len, test_len))


def main():
    root_folder = 'dataset_final_sorted'
    language_list = ['c', 'cpp', 'py']

    # for language in language_list:
    #     file_name = language + '_divided.jsonl'
    #     dump_files_by_language_from_subfolder(root_folder=root_folder, language=language, output_dir=file_name)

    for language in language_list:
        file_name = language + '_divided.jsonl'
        split_dataset(file_name, language)


def test():
    """
    仅用作测试各种特殊情况
    """
    # file_path = 'dataset_final_sorted/CWE-264/c/bad_2399_0'
    file_path = 'cpp_divided.jsonl'
    output_dir = 'test.json'
    func('c', file_path, output_dir)


if __name__ == '__main__':
    main()
    # test()
