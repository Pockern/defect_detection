import re

def remove_comments(code):
    # 定义正则表达式模式，匹配 Python 单行注释和行尾注释
    pattern = r'#.*?$|#$'
    # 使用 re.sub() 函数替换匹配的注释为空字符串
    cleaned_code = re.sub(pattern, '', code, flags=re.MULTILINE)
    return cleaned_code

# 示例代码，包含单行和行尾注释
code = """
pid = context.project_id            msg = _(\"Quota exceeded for %(pid)s, tried to set \"                    \"%(num_metadata)s metadata properties\") % locals()            LOG.warn(msg)            raise exception.QuotaError(code=\"MetadataLimitExceeded\")        # Because metadata is stored in the DB, we hard-code the size limits        # In future, we may support more variable length strings, so we act        #  as if this is quota-controlled for forwards compatibility        for k, v in metadata.iteritems():
"""

# 去除注释后的代码
cleaned_code = remove_comments(code)
print(cleaned_code)