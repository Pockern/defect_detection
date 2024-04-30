import json
import os
import re
import subprocess
import warnings
import random

from tree_sitter import Language, Parser

warnings.filterwarnings('ignore', category=FutureWarning)

# Language.build_library(
#     'parser/languages.so',
#     [
#         'vendor/tree-sitter-c',
#         'vendor/tree-sitter-cpp',
#         'vendor/tree-sitter-python',
#         'vendor/tree-sitter-javascript',
#         'vendor/tree-sitter-java'
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
        pattern = r'//.*?$|/\*.*?\*/|/\*[\s\S]*?\*/'
    elif language == 'py':
        pattern = r'#.*?$|#$|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|""".*?"""'
        language = 'python'
    elif language == 'js':
        pattern = r'//.*?$|/\*.*?\*/|/\*[\s\S]*?\*/|\*.*?$'
        language = 'javascript'
    elif language == 'php':
        pattern = r'//.*?$|/\*.*?\*/|/\*[\s\S]*?\*/|\*.*?$|#.*?$'
    elif language == 'java':
        pattern = r'//.*?$|/\*.*?\*/|/\*[\s\S]*?\*/|\*.*?$|/\*\*.*?\*/'

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
    elif language == 'js':
        language = 'javascript'

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
                     "function_declaration", "call", "local_function_statement", "class_specifier",
                     
                     "compound_statement", "struct_specifier", "preproc_ifdef", 
                     "namespace_definition", 
                    #  "block",
                     "class_definition", "decorated_definition",
                     ]

    functions = []
    functions_starts = []
    functions_ends = []

    if language == 'java':
        for node in tree.root_node.children:
            if node.type == 'method_declaration':
                functions.append(code[node.start_byte:node.end_byte].decode())
                functions_starts.append(node.start_point[0]+1)
                functions_ends.append(node.end_point[0]+1)
            elif node.type == 'class_declaration':
                for part in node.children:
                    if part.type == 'class_body':
                        for child in part.children:
                            if child.type == 'method_declaration':
                                functions.append(code[child.start_byte:child.end_byte].decode())
                                functions_starts.append(child.start_point[0]+1)
                                functions_ends.append(child.end_point[0]+1)

    elif language == 'cpp':
        for node in tree.root_node.children:
            if node.type == 'class_specifier':
                functions.append(code[node.start_byte:node.end_byte].decode())
                functions_starts.append(node.start_point[0]+1)
                functions_ends.append(node.end_point[0]+1)
            elif node.type == 'function_definition':
                functions.append(code[node.start_byte:node.end_byte].decode())
                functions_starts.append(node.start_point[0]+1)
                functions_ends.append(node.end_point[0]+1)

    elif language == 'javascript':
        for node in tree.root_node.children:
            if node.type == 'expression_statement':
                for part in node.children:
                    if part.type == 'assignment_expression':
                        if 'function_expression' in [child.type for child in part.children]:
                            functions.append(code[part.start_byte:part.end_byte].decode())
                            functions_starts.append(part.start_point[0]+1)
                            functions_ends.append(part.end_point[0]+1)
                    elif part.type == 'call_expression':
                        for child in part.children:
                            if child.type == 'member_expression':
                                if 'identifier' in [c.type for c in child.children]:
                                    functions.append(code[part.start_byte:part.end_byte].decode())
                                    functions_starts.append(part.start_point[0]+1)
                                    functions_ends.append(part.end_point[0]+1)
                        
            elif node.type == 'lexical_declaration':
                for part in node.children:
                    if part.type == 'variable_declarator':
                        if 'function_expression' in [child.type for child in part.children]:
                            functions.append(code[part.start_byte:part.end_byte].decode())
                            functions_starts.append(part.start_point[0]+1)
                            functions_ends.append(part.end_point[0]+1)

            elif node.type == 'if_statement':
                functions.append(code[node.start_byte:node.end_byte].decode())
                functions_starts.append(node.start_point[0]+1)
                functions_ends.append(node.end_point[0]+1)

    else:
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
            if patches_divided_starts[j] >= functions_starts[i]-5 and patches_divided_starts[j] <= functions_ends[i]+5:
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
            print('collect a pair of file: {}'.format(object_dict['cwe'] + '/' + language + '/' + object_dict['cwe_id']))

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


def split_dataset(file_name, language, seed, limit_low, limit_high):
    output_dir = language
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # parameter init
    # random.seed(seed)
    train_data_file = os.path.join(output_dir, 'train.jsonl')
    valid_data_file = os.path.join(output_dir, 'valid.jsonl')
    test_data_file = os.path.join(output_dir, 'test.jsonl')
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    with open(file_name, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    # random.shuffle(data)
    good_files = []
    bad_files = []
    for cnt, object in enumerate(data):
        bad_tag = False
        good_tag = False
        if object['before'] == '404: Not Found' and object['after'] != '404: Not Found':
            good_tag = True
        elif object['before'] != '404: Not Found' and object['after'] == '404: Not Found':
            bad_tag = True


        if cnt % 2 == 0 or bad_tag == True:
            file_idx = object['cwe'] + '/bad'+object['cwe_id']
            functions = object['functions_before_patches']
            if len(functions) >= limit_low and len(functions) <= limit_high:
                file_code = re.sub(r"[\n\t]", "", object['before'])     # 除去 \n\t
                for function in functions:
                    function['function_code'] = re.sub(r"[\n\t]", "", function['function_code'])
                language = object['language']
                file_label = 1
                
                bad_files.append(DatasetEntry(file_idx, functions, file_code, language, file_label).to_dict())

        # -------------good -----------------------------------s
        if cnt % 2 != 0 or good_tag == True:
            file_idx = object['cwe'] + '/good'+object['cwe_id']
            functions = object['functions_after_patches']
            if len(functions) >= limit_low and len(functions) <= limit_high:
                file_code = re.sub(r"[\n\t]", "", object['after'])      # 除去 \n\t
                for function in functions:
                    function['function_code'] = re.sub(r"[\n\t]", "", function['function_code'])
                language = object['language']
                file_label = 0
                
                good_files.append(DatasetEntry(file_idx, functions, file_code, language, file_label).to_dict())

    
    bad_len = len(bad_files)
    good_len = len(good_files)
    random.shuffle(bad_files)
    # random.shuffle(good_files)

    train_good_len = int(good_len * train_ratio)
    valid_good_len = int(good_len * valid_ratio)
    test_good_len = good_len - train_good_len - valid_good_len

    train_bad_len = int(bad_len * train_ratio)
    valid_bad_len = int(bad_len * valid_ratio)
    test_bad_len = bad_len - train_bad_len - valid_bad_len

    train_data = good_files[:train_good_len]
    valid_data = good_files[train_good_len:train_good_len+valid_good_len]
    test_data = good_files[train_good_len+valid_good_len:]

    train_data.extend(bad_files[:train_bad_len])
    valid_data.extend(bad_files[train_bad_len:train_bad_len+valid_bad_len])
    test_data.extend(bad_files[train_bad_len+valid_bad_len:])

    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)

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

    print('{} -- train dataset: {}, valid dataset: {}, test dataset: {}'.format(language, len(train_data), len(valid_data), len(test_data)))


# def limit_functions(file_name, output_dir, limit_low, limit_high):
#     js_objects = []
#     with open(file_name, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             js_objects.append(data)

#     dest_js = []
#     for object in js_objects:
#         functions = object['functions'][0]
#         # magic number by test(87 for single test)
#         if len(functions) <= limit_high and len(functions) >= limit_low:
#             dest_js.append(object)

#     with open(output_dir, 'w') as f:
#         for data in dest_js:
#             json.dump(data, f)
#             f.write('\n')
#     print('{}: from {} collect {}'.format(file_name, len(js_objects), len(dest_js)))


def main():
    root_folder = 'dataset_final_sorted'
    language_list = ['c', 'cpp', 'py', 'js', 'java']

    # for language in language_list:
    #     file_name = language + '_divided.jsonl'
    #     output_dir = 'dump_files'
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     output_dir = os.path.join(output_dir, file_name)
    #     dump_files_by_language_from_subfolder(root_folder=root_folder, language=language, output_dir=output_dir)

    # for language in language_list:
    #     file_name = 'dump_files/' + language + '_divided.jsonl'
    #     output_dir = 'func_files/'
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     output_dir = output_dir + language + '_divided.jsonl'
    #     func(language, file_name, output_dir)

    for language in language_list:
        file_name = language + '_divided.jsonl'
        file_name = os.path.join('func_files', file_name)
        split_dataset(file_name, language, seed=123456, limit_low=1, limit_high=20)


def func_for_test(language, divided_file_name, dest_file_name):
    # open a jsonl file that each line represent a json object
    with open(divided_file_name, "r", encoding = "utf-8") as r:
        content = r.readlines()
        # traverse each object of jsonl
        for idx in range(len(content)):
            object_dict = json.loads(content[idx])
            file_idx = object_dict['cwe'] + '/' + object_dict['language'] + '/' + object_dict['cwe_id']
            if file_idx == dest_file_name:
                
                code_before_patched =  object_dict['before']['file_code']

                patches = object_dict['patches']
                with open('patches.temp', 'w') as f:
                    f.write(patches)

                pattern = re.compile(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@')
                matches = pattern.findall(patches)
                patch_divided_temp = patches.split('@@')
                patches_divided = []
                patches_divided_starts = []
                cnt = 1
                for match in matches:
                    old_start, old_lines, new_start, new_lines = int(match[0]), match[1], int(match[2]), match[3]
                    patches_divided_starts.append(old_start)
                    patches_divided.append( '@@ ' + patch_divided_temp[cnt] + ' @@' + patch_divided_temp[cnt+1])
                    cnt += 2

                functions, functions_starts, functions_ends = slice(code_before_patched, language)
                with open('functions.temp', 'w') as f:
                    for function in functions:
                        f.write(remove_comment(function, object_dict['language']))
                        f.write('\n'+'-'*20 + '\n')

                functions_label = get_label_of_functions_by_patches(functions_starts, functions_ends, patches_divided_starts)

def test():
    """
    仅用作测试各种特殊情况
    """
    file_path = 'dump_files/py_divided.jsonl'
    dest = 'CWE-287/py/_4331_0'

    func_for_test('py', file_path, dest)


if __name__ == '__main__':
    main()
    # test()
