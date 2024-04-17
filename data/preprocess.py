import json
import os
from tree_sitter import Language, Parser
import re
import subprocess

# 构筑语言模块
# Language.build_library(
#     'parser/languages.so',
#     [
#         'vendor/tree-sitter-c',
#         'vendor/tree-sitter-cpp',
#         'vendor/tree-sitter-python'
#     ]
# )

def slice(code, language):
    """
    divide code into multiple functions
    :param code: dest_code
    :param language: the PL of code
    :return: list of divided functions, list of function_starts, list of functions_ends
    """
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


def func(language, file_name, write_name):  
    count_all = 0
    count_not = 0
    # open a jsonl file that each line represent a json object
    with open(file_name, "r", encoding = "utf-8") as r:
        content = r.readlines()
        # traverse each object
        for idx in range(len(content)):
            record_dict = json.loads(content[idx])
            code_before_patched = record_dict['before']
            code_after_patched = record_dict['after']
            patches = record_dict['patches']
            pattern = re.compile(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@')
            matches = pattern.findall(patches)
            patch_divided_temp = patches.split('@@')

            patches_divided = []
            patches_divided_starts = []
            # count from 1: actually ignore 1st part of the diff file, which is useless in xxx.diff
            cnt = 1
            for match in matches:
                old_start, old_lines, new_start, new_lines = match[0], match[1], int(match[2]), match[3]
                patches_divided_starts.append(new_start)
                # incorporate a wholt patch
                patches_divided.append( '@@ ' + patch_divided_temp[cnt] + ' @@' + patch_divided_temp[cnt+1])
                cnt += 2
            
            # print(record_dict['cwe'] + record_dict['cwe_id'])
            # print(patches_divided[0])
            # return

            functions, functions_starts, functions_ends = slice(code_before_patched['code'], language)

            print(functions)

            return

            # functions_patchs = []
            # functions_patchs_remain = []
            # patches_divided_label = [0] * len(patches_divided)
            # # for each function
            # for i in range(len(functions_starts)):
            #     function_temp = ''
            #     patch_temp = ''
            #     # for each patch
            #     for j in range(len(patches_divided_starts)):
            #         max = 0
            #         # patch[j] close to this func[i]
            #         if patches_divided_starts[j] >= functions_starts[i]-5 and patches_divided_starts[j] <= functions_ends[i]+5:
            #             # max: the max len of function in file
            #             if functions_ends[i] - functions_starts[i] > max:
            #                 max = functions_ends[i] - functions_starts[i]
            #                 function_temp = functions[i]
            #             # union patches close to func?
            #             if patches_divided_label[j] == 0:
            #                 patch_temp = patch_temp + patches_divided[j]
            #                 patches_divided_label[j] = 1

            #     if function_temp != '' and patch_temp != '':
            #         function_patch = {}
            #         function_patch['function'] = function_temp
            #         function_patch['patch'] = patch_temp
            #         functions_patchs.append(function_patch)
            #         print(function_patch)
            #         return


    #         # details include attribute like: code, patch
    #         details = record_dict['details']
    #         for idx1, detail in enumerate(details):
    #             code = detail['code']
    #             patch = detail['patch']
    #             # get some diff parameters
    #             pattern = re.compile(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@')
    #             matches = pattern.findall(patch)
    #             patch_small_temp = patch.split('@@')
                
    #             patch_small = []
    #             patch_new_starts = []

    #             # 遍历每个匹配并输出修改的代码内容
    #             # count from 1: actually ignore the head of the file, which is useless in xxx.diff
    #             cnt = 1
    #             for match in matches:
    #                 old_start, old_lines, new_start, new_lines = match[0], match[1], int(match[2]), match[3]
    #                 patch_new_starts.append(new_start)
    #                 # patch_small includes each part of new patch by (id? as matches)
    #                 patch_small.append( '@@ ' + patch_small_temp[cnt] + ' @@' + patch_small_temp[cnt+1])
    #                 cnt += 2

    #             # divide file-level into function-level
    #             functions, functions_starts, functions_ends = slice(code, language)

    #             functions_patchs = []
    #             functions_patchs_remain = []
    #             patch_small_flag = [0] * len(patch_small)
    #             # for each function in file
    #             for i in range(len(functions_starts)):
    #                 function_temp = ''
    #                 patch_temp = ''
    #                 # for each patch in file
    #                 for j in range(len(patch_new_starts)):
    #                     max = 0
    #                     # patch[j] close to this func[i]
    #                     if patch_new_starts[j] >= functions_starts[i]-5 and patch_new_starts[j] <= functions_ends[i]+5:
    #                         # max: the max len of function in file
    #                         if functions_ends[i] - functions_starts[i] > max:
    #                             max = functions_ends[i] - functions_starts[i]
    #                             function_temp = functions[i]
    #                         # union patches close to func?
    #                         if patch_small_flag[j] == 0:
    #                             patch_temp = patch_temp + patch_small[j]
    #                             patch_small_flag[j] = 1

    #                 if function_temp != '' and patch_temp != '':
    #                     function_patch = {}
    #                     function_patch['function'] = function_temp
    #                     function_patch['patch'] = patch_temp
    #                     functions_patchs.append(function_patch)

    #             # loop num: sizeof(patches)
    #             for j in range(len(patch_small_flag)):
    #                 if patch_small_flag[j] == 0:
    #                     functions_patchs_remain.append(patch_small[j])
    #             record_dict['details'][idx1]['functions_patchs'] = functions_patchs
    #             record_dict['details'][idx1]['functions_patchs_remain'] = functions_patchs_remain
    #             # record_dict['details'][idx1]['functions_starts'] = functions_starts
    #             # record_dict['details'][idx1]['functions_ends'] = functions_ends
    #             # record_dict['details'][idx1]['patch_starts'] = patch_new_starts
    #             if record_dict['details'][idx1]['file_language'] == language:
    #                 count_all += 1
    #                 if functions_patchs == []:
    #                     count_not += 1

    #         with open(write_name, "a", encoding = "utf-8") as rf:
    #             rf.write(json.dumps(record_dict) + '\n')
    # print(count_not)
    # print(count_all)


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
            "before": self.before,              # ['code', 'file_label']
            "after": self.after,
            "patches": self.patches,
            "language": self.language,
            "cwe": self.cwe,
            "cwe_id": self.cwe_id
        }


def copy_files_by_language_from_subfolder(root_folder, language, output_dir):
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
                        before = {}
                        file_before_path = os.path.join(sub_root, file)
                        with open(file_before_path, 'r') as f:
                            content = f.read()
                            before['code'] = content
                            before['file_label'] = 1

                        # preprocess file with patches
                        after = {}
                        file_after_path = file_before_path.replace('bad', 'good')
                        with open(file_after_path, 'r') as f:
                            content = f.read()
                            after['code'] = content
                            after['file_label'] = 0

                        # get patches
                        file_patch_path = file_after_path.replace('good', '') + '.diff'
                        with open(file_patch_path, 'w') as f:
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


if __name__ == '__main__':
    root_folder = './dataset_final_sorted'
    # for language in ['c', 'cpp', 'py']:
    for language in ['cpp']:
        output_dir = language + '_with_patches.jsonl'
        copy_files_by_language_from_subfolder(root_folder=root_folder, language=language, output_dir=output_dir)

    # language = 'c'
    # file_name = 'c_with_patches.jsonl'

    # --------------test-------------------------
    # with open(file_name, 'r') as f:
    #     line = f.readline()
    #     js = json.loads(line)

    # print(js)
    # --------------------------------------------
        
    # output = 'c_divided.jsonl'
    # func(language, file_name, output)
