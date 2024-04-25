import json
import os
import re
import subprocess
import warnings
import random

from tree_sitter import Language, Parser

warnings.filterwarnings('ignore', category=FutureWarning)

# 构筑语言模块
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


class FunctionEntry:
    def __init__(self, function_idx, function_code, function_label, bag_label):
        self.function_idx = function_idx
        self.function_code = function_code
        self.function_label = function_label
        self.bag_label = bag_label
    
    def to_dict(self):
        return {
            "function_idx": self.function_idx, 
            "function_code": self.function_code,
            "function_label": self.function_label,
            "bag_label": self.bag_label
        }
    

class DatasetEntry:
    def __init__(self, file_idx, functions, file_code, language, file_label):
        self.file_idx = file_idx
        self.functions = functions,
        self.file_code = file_code,
        self.language = language,
        self.file_label = file_label

    def to_dict(self):
        return {
            "file_idx": self.file_idx,
            "functions": self.functions,
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


def slice(code, language):
    """
    divide code into multiple functions
    :param code: dest_code
    :param language: the PL of code
    :return: list of divided functions, list of function_starts, list of functions_ends
    """
    if language == 'py':
        language = 'python'

    # 加载语言模块
    LANGUAGE = Language('parser/languages.so', language)

    parser = Parser()
    parser.set_language(LANGUAGE)
    code = code.encode()
    tree = parser.parse(code)

    # C/CPP: function_declarator
    # Java: class_declaration, method_declaration
    # python: function_definition, call
    # js: function_declaration

    function_type = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                     "function_declaration", "call", "local_function_statement", "class_specifier"
                     
                     "compound_statement", "struct_specifier", "preproc_ifdef", 
                     "namespace_definition", 
                    #  "block",
                     "class_definition", "decorated_definition",
                     ]
    functions = []
    functions_starts = []
    functions_ends = []

    for node in tree.root_node.children:
        if node.type in function_type:
            functions.append(code[node.start_byte:node.end_byte].decode())
            functions_starts.append(node.start_point[0]+1)
            functions_ends.append(node.end_point[0]+1)

    return functions, functions_starts, functions_ends


def get_label_of_functions_by_patches(functions_starts, functions_ends, patches_divided_starts):
    """
    give label of vulnerablity function (only for code before patched)
    """
    functions_label = [0] * len(functions_starts)
    # for each function
    for i in range(len(functions_starts)):
        # for each patch
        for j in range(len(patches_divided_starts)):
            # patch[j] close to this func[i]
            if patches_divided_starts[j] >= functions_starts[i]-1 and patches_divided_starts[j] <= functions_ends[i]+1:
                # union patches close to func?
                if functions_label[i] == 0:
                    functions_label[i] = 1

    return functions_label


def func(language, file_name, output_dir):
    # open a jsonl file that each line represent a json object
    with open(file_name, "r", encoding = "utf-8") as r:
        content = r.readlines()
        output = []
        # traverse each object of jsonl
        for idx in range(len(content)):
            object_dict = json.loads(content[idx])

            code_before_patched =  object_dict['before']['file_code']
            # code_before_patched =  remove_comment(object_dict['before']['file_code'], object_dict['language'])
            # with open('before_patched.c', 'w') as f:
            #     f.write(code_before_patched)

            code_after_patched =  object_dict['after']['file_code']
            # code_after_patched =  remove_comment(object_dict['after']['file_code'], object_dict['language'])
            # with open('after_patched.c', 'w') as f:
            #     f.write(code_after_patched)

            # divide patches
            patches = object_dict['patches']
            # with open('patches.c', 'w') as f:
            #     f.write(patches)

            pattern = re.compile(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@')
            matches = pattern.findall(patches)
            patch_divided_temp = patches.split('@@')
            patches_divided = []
            patches_divided_starts = []
            # count from 1: actually ignore 1st part of the diff file, which is useless in xxx.diff
            cnt = 1
            for match in matches:
                old_start, old_lines, new_start, new_lines = int(match[0]), match[1], int(match[2]), match[3]
                patches_divided_starts.append(old_start)
                # incorporate a whole patch
                patches_divided.append( '@@ ' + patch_divided_temp[cnt] + ' @@' + patch_divided_temp[cnt+1])
                cnt += 2
        
            # divide functions
            # bad
            functions, functions_starts, functions_ends = slice(code_before_patched, language)
            functions_label = get_label_of_functions_by_patches(functions_starts, functions_ends, patches_divided_starts)
            functions_object_list = [FunctionEntry(idx, remove_comment(func, language), label, 1).to_dict() for idx, (func, label) in enumerate(zip(functions, functions_label))]
            object_dict['functions_before_patches'] = functions_object_list
            object_dict['before'] = remove_comment(code_before_patched, language)
            # good
            functions, functions_starts, functions_ends = slice(code_after_patched, language)
            functions_label = [0] * len(functions)
            functions_object_list = [FunctionEntry(idx, remove_comment(func, language), label, 0).to_dict() for idx, (func, label) in enumerate(zip(functions, functions_label))]
            object_dict['functions_after_patches'] = functions_object_list
            object_dict['after'] = remove_comment(code_after_patched, language)

            output.append(object_dict)
            print('divide a pair of file: {}'.format(object_dict['cwe'] + '/' + language + '/' + object_dict['cwe_id']))

    with open(output_dir, 'w') as f:
        for data in output:
            json.dump(data, f)
            f.write('\n')


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
        # json.dump(js, f)
    print('collect {} pairs of good/bad files from {}'.format(file_idx+1, language))


def split_dataset(file_name, language, seed):
    output_dir = language
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # parameter init
    random.seed(seed)
    train_data_file = os.path.join(output_dir, 'train.jsonl')
    valid_data_file = os.path.join(output_dir, 'valid.jsonl')
    test_data_file = os.path.join(output_dir, 'test.jsonl')
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    with open(file_name, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    # random.shuffle(data)
    files = []
    for cnt, object in enumerate(data):
        file_idx = object['cwe'] + '/bad'+object['cwe_id']
        functions = object['functions_before_patches']
        file_code = object['before']
        # 除去 \n\t
        for function in functions:
            function['function_code'] = re.sub(r"[\n\t]", "", function['function_code'])
        language = object['language']
        file_label = 1
        if len(functions) > 0 and cnt % 2 == 0:
            files.append(DatasetEntry(file_idx, functions, file_code, language, file_label).to_dict())
        # if len(functions) > 0:
        #     files.append(DatasetEntry(file_idx, functions, language, file_label).to_dict())

        # -------------good -----------------------------------s
        file_idx = file_idx.replace('bad', 'good')
        functions = object['functions_after_patches']
        file_code = object['after']
        # 除去 \n\t
        for function in functions:
            function['function_code'] = re.sub(r"[\n\t]", "", function['function_code'])
        language = object['language']
        file_label = 0
        if len(functions) > 0 and cnt % 2 != 0:
            files.append(DatasetEntry(file_idx, functions, file_code, language, file_label).to_dict())
        # if len(functions) > 0:
        #     files.append(DatasetEntry(file_idx, functions, language, file_label).to_dict())

    
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


def limit_functions(file_name, output_dir, limit_low, limit_high):
    js_objects = []
    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            js_objects.append(data)

    dest_js = []
    for object in js_objects:
        functions = object['functions'][0]
        # magic number by test(87 for single test)
        if len(functions) <= limit_high and len(functions) >= limit_low:
            dest_js.append(object)

    with open(output_dir, 'w') as f:
        for data in dest_js:
            json.dump(data, f)
            f.write('\n')
    print('{}: from {} collect {}'.format(file_name, len(js_objects), len(dest_js)))


def main():
    root_folder = 'dataset_final_sorted'
    language_list = ['c', 'py', 'js', 'java', 'php']

    # for language in language_list:
    #     file_name = language + '_divided.jsonl'
    #     dump_files_by_language_from_subfolder(root_folder=root_folder, language=language, output_dir=file_name)

    for language in language_list:
        file_name = language + '_divided.jsonl'
        output = file_name
        func(language, file_name, output)

    # for language in language_list:
    #     file_name = language + '_divided.jsonl'
    #     split_dataset(file_name, language, 123456)

    # for language in language_list:
    #     for file in ['train', 'test', 'valid']:
    #         file_name = os.path.join(language, file + '.jsonl')
    #         output_dir = file_name
    #         limit_functions(file_name, output_dir, 1, 64)

    # for language in language_list:
    #     for file in ['train.jsonl']:
    #         file_name = os.path.join(language, file)
    #         output_dir = file_name
    #         limit_functions(file_name, output_dir, 1, 15)


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
