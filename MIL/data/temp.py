from tree_sitter import Language, Parser  # 解析器库
from graphviz import Digraph, Source  # 绘图库


LANGUAGE = Language('parser/languages.so', 'c')

# 创建 Tree-sitter 解析器
parser = Parser()
parser.set_language(LANGUAGE)

# 要解析的 C 语言源代码
code = b"""
#include <stdio.h>

int main()
{
    int a = 0;
}
"""

# code = code.encode()
tree = parser.parse(code)


dot = Digraph()

# 辅助函数，用于构建图
def build_graph(node, parent_label=None):
    current_label = node.type
    dot.node(current_label)
    if parent_label is not None:
        dot.edge(parent_label, current_label)
    for child in node.children:
        build_graph(child, current_label)

# 从根节点开始构建图
build_graph(tree.root_node)

# 渲染图
dot.render('ast', format='png', cleanup=True)