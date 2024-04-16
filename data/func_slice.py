from tree_sitter import Language, Parser
import json
import re

# Language.build_library(
#     'parser/languages.so',
#     [
#         'vendor/tree-sitter-c',
#         'vendor/tree-sitter-cpp',
#         'vendor/tree-sitter-python'
#     ]
# )

count_all = 0
count_not = 0

def slice(code, language):
    # 加载 JavaScript 语言模块
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

# ----------------------- just for test-------------------------------
def func1(file_name):
    with open(file_name, 'r') as r:
        patch = r.read()
        pattern = re.compile(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@')
        matches = pattern.findall(patch)
        print(matches)
        patch_small_temp = patch.split('@@')
        
        patch_small = []
        patch_new_starts = []

        # 遍历每个匹配并输出修改的代码内容
        # count from 1: actually ignore the head of the file
        cnt = 1
        for match in matches:
            old_start, old_lines, new_start, new_lines = match[0], match[1], int(match[2]), match[3]
            patch_new_starts.append(new_start)
            patch_small.append( '@@ ' + patch_small_temp[cnt] + ' @@' + patch_small_temp[cnt+1])
            cnt += 2

        print(patch_small[1])
# -------------------------------------------------------------------


def func(language, file_name, write_name):  
    global count_all, count_not
    count_all = 0
    count_not = 0
    # open a json meta-file that each line represent a json file
    with open(file_name, "r",encoding = "utf-8") as r:
        content = r.readlines()
        for idx in range(len(content)):
            # load each json file
            record_dict = json.loads(content[idx])
            # details include attribute like: code, patch
            details = record_dict['details']
            for idx1, detail in enumerate(details):
                code = detail['code']
                patch = detail['patch']
                # get some diff parameters
                pattern = re.compile(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@')
                matches = pattern.findall(patch)
                patch_small_temp = patch.split('@@')
                
                patch_small = []
                patch_new_starts = []

                # 遍历每个匹配并输出修改的代码内容
                # count from 1: actually ignore the head of the file, which is useless in xxx.diff
                cnt = 1
                for match in matches:
                    old_start, old_lines, new_start, new_lines = match[0], match[1], int(match[2]), match[3]
                    patch_new_starts.append(new_start)
                    # patch_small includes each part of new patch by (id? as matches)
                    patch_small.append( '@@ ' + patch_small_temp[cnt] + ' @@' + patch_small_temp[cnt+1])
                    cnt += 2

                # divide file-level into function-level
                functions, functions_starts, functions_ends = slice(code, language)

                functions_patchs = []
                functions_patchs_remain = []
                patch_small_flag = [0] * len(patch_small)
                # for each function in file
                for i in range(len(functions_starts)):
                    function_temp = ''
                    patch_temp = ''
                    # for each patch in file
                    for j in range(len(patch_new_starts)):
                        max = 0
                        # patch[j] close to this func[i]
                        if patch_new_starts[j] >= functions_starts[i]-5 and patch_new_starts[j] <= functions_ends[i]+5:
                            # max: the max len of function in file
                            if functions_ends[i] - functions_starts[i] > max:
                                max = functions_ends[i] - functions_starts[i]
                                function_temp = functions[i]
                            # union patches close to func?
                            if patch_small_flag[j] == 0:
                                patch_temp = patch_temp + patch_small[j]
                                patch_small_flag[j] = 1

                    if function_temp != '' and patch_temp != '':
                        function_patch = {}
                        function_patch['function'] = function_temp
                        function_patch['patch'] = patch_temp
                        functions_patchs.append(function_patch)

                # loop num: sizeof(patches)
                for j in range(len(patch_small_flag)):
                    if patch_small_flag[j] == 0:
                        functions_patchs_remain.append(patch_small[j])
                record_dict['details'][idx1]['functions_patchs'] = functions_patchs
                record_dict['details'][idx1]['functions_patchs_remain'] = functions_patchs_remain
                # record_dict['details'][idx1]['functions_starts'] = functions_starts
                # record_dict['details'][idx1]['functions_ends'] = functions_ends
                # record_dict['details'][idx1]['patch_starts'] = patch_new_starts
                if record_dict['details'][idx1]['file_language'] == language:
                    count_all += 1
                    if functions_patchs == []:
                        count_not += 1

            with open(write_name, "a", encoding = "utf-8") as rf:
                rf.write(json.dumps(record_dict) + '\n')
    print(count_not)
    print(count_all)
            

def main():

    # ----------------------------------------------------------------------------
    # test for myself
    # language = "c"
    # with open('./c/bad_1_0') as f:
    #     code = f.read()
    # functions, functions_starts, functions_ends = slice(code, language=language)
    # print(functions[-1])
    # print(functions_starts[-1])
    # print(functions_ends[-1])

    filename = 'temp.diff'
    func1(file_name=filename)
    # ----------------------------------------------------------------------------

    # language = "c"
    # file_name = '/data/xcwen/Challenge/Method/TreeSitter/language/merge_C.jsonl' 
    # write_name = '/data/xcwen/Challenge/Method/TreeSitter/language_new/merge_C.jsonl'  
    # func(language, file_name, write_name)
    
    # language = "cpp"
    # file_name = '/data/xcwen/Challenge/Method/TreeSitter/language/merge_C++.jsonl'  
    # write_name = '/data/xcwen/Challenge/Method/TreeSitter/language_new/merge_C++.jsonl'  
    # func(language, file_name, write_name)

    # language = "java"
    # file_name = '/data/xcwen/Challenge/Method/TreeSitter/language/merge_Java.jsonl'  
    # write_name = '/data/xcwen/Challenge/Method/TreeSitter/language_new/merge_Java.jsonl'  
    # func(language, file_name, write_name)

    # language = "python"
    # file_name = '/data/xcwen/Challenge/Method/TreeSitter/language/merge_Python.jsonl' 
    # write_name = '/data/xcwen/Challenge/Method/TreeSitter/language_new/merge_Python.jsonl'  
    # func(language, file_name, write_name)


if __name__ == "__main__":
    main()