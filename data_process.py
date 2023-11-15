import sys
import os
from tqdm import tqdm
import re

root_path = os.path.dirname(__file__)


def punctuation_split(source):
    source = source.replace('\\"', '"')
    source = source.replace("_", " ").replace("\"", " \" ").replace("\'", " \' ").replace(".", " . ").replace(":",
                                                                                                              " : ").replace(
        "\\", " \\ ").replace("|", " | ").replace("^", " ^ ").replace("/", " / ").replace("`", " ` ").replace(",",
                                                                                                              " , ").replace(
        "%", " % ")
    source = source.replace("!", " ! ").replace("{", " { ").replace("}", " } ").replace("(", " ( ").replace(")",
                                                                                                            " ) ").replace(
        "-", " - ").replace("<", " < ").replace(">", " > ").replace("*", " * ").replace("#", " # ").replace("[",
                                                                                                            " [ ").replace(
        "]", " ] ").replace(";", " ; ")

    while (re.search("  ", source)):
        source = source.replace("  ", " ")
    while (re.search("- -", source)):
        source = source.replace("- -", "--")
    source = source.replace("e . g .", "e.g.").replace("didn t", "didn't").replace("> =", ">=").replace("< =",
                                                                                                        "<=").replace(
        "* * kwargs", "**kwargs").replace("* args", "*args").replace("% s", "%s").replace("% d", "%d").replace(". .",
                                                                                                               "..").replace(
        ". . .", "...").replace(". .", "..").replace("< p >", "")
    source = source.replace("\\ n", "\\n").replace("` `", '"').replace("`", "'").replace("doesn ' t",
                                                                                         "doesn't").replace("don ' t",
                                                                                                            "don't").replace(
        'r " " " ', '').replace("i . i . d .", "i.i.d").replace("- >", "->").replace("r ' ' ' ", '').replace(
        " / @@api / ", '').replace("\\ r ", '').replace("> > >", '>>>')
    return source.lower()


# def is_special(source):
#     for string in source:
#         for ch in string:
#             if (u'\u4e00' <= ch <= u'\u9fff'):
#                 return True

def is_special(source):
    if (re.search(r".. note : : experimental", source) or re.search("http", source) or re.search(r'[\u4e00-\u9fff]',
                                                                                                 source) or re.search(
            r"\\ u....", source) \
        or re.search("versionadded", source) or re.search("versionchanged", source) or re.search(":: rtype",
                                                                                                 source) or re.search(
                ".. note", source)) \
            or re.search("==================", source) or re.search("-----------------", source) or re.search(
        "< here >", source) or re.search("non - javadoc", source) \
            or re.search("@deprecated", source) or re.search("< b >", source) or re.search(r"\* \* \* \* \* \* \* \* ",
                                                                                           source) or re.search("< b >",
                                                                                                                source) or re.search(
        "< ! -- begin - user - doc -->", source):
        return True


def del_comment(code):
    i = 0
    for _ in range(len(code)):
        if (re.search('#', code[i]) or re.search('"""(.*)', code[i]) or re.search(r'\r', code[i]) or re.search(
                r'\"\"\"', code[i]) or re.search(r"\'\'\'", code[i]) or re.search("'''(.*)", code[i]) or re.search(
                r'[\u4e00-\u9fff]', code[i]) or re.search("\\\n", code[i]) or re.search("//", code[i])):
            del code[i]
            i -= 1
        code[i] = punctuation_split(code[i])
        i += 1
    return code


def del_punctuation(source):
    i = 0
    for _ in range(len(source)):
        if (re.search(":", source[i])):
            del source[i]
            i -= 1
        i += 1
    return source


def find_code(source):
    pos1 = re.search('"code_tokens":', source).span()[1] + 1
    pos2 = re.search('"docstring":', source).span()[0] - 2
    code = eval(source[pos1:pos2])
    code = del_comment(code)
    if (len(code) > 300):
        return None
    code = ' '.join(code)
    code = punctuation_split(code)
    return code


def find_codes(source):
    pos1 = re.search('"code":', source).span()[1] + 2
    pos2 = re.search('"code_tokens":', source).span()[0] - 3
    code = source[pos1:pos2]
    return code


def find_title(source):
    pos1 = re.search('"docstring":', source).span()[1] + 2
    pos2 = re.search('"docstring_tokens":', source).span()[0] - 3
    title = source[pos1:pos2]
    # title = train(title)
    # title = del_punctuation(title)
    # title = ' '.join(title)
    if (is_special(title)):
        return None
    title = title.replace(":", "")
    title = title.replace("{", "").replace("}", "")
    title = punctuation_split(title)
    title = title.replace("'", "")
    pos1 = title.find("\\n")
    pos2 = title.find(".")
    try:
        pos3 = re.search(r'----*', source).span()[0] - 1
    except AttributeError:
        if (pos2):
            title = title[:pos2]
        elif (pos1):
            title = title[:pos1]
    else:
        if (pos2 == -1):
            title = title[:pos3]

        else:
            if (pos2 > pos3):
                title = title[:pos3]
            else:
                title = title[:pos2]

    if (len(title) <= 20 or is_special(title)):
        return None
    pos = title.find(',')
    if (pos != -1):
        title = title[:pos]
    pos = title.find('#')
    if (pos != -1):
        title = title[:pos]
    while (title[-1] == ' '):
        if (len(title) <= 10):
            return None
        title = title[:-1]

    if (title[-1] != '.'):
        title = title + ' .'
    while (title.find("\\n") != -1):
        title = title.replace("\\n", "")
    while (title[0] == ' '):
        title = title[1:]
    while (re.search("  ", title)):
        title = title.replace("  ", " ")
    while (re.search(r"/ / ", title)):
        title = title.replace(r"/ / ", "")
    return title


if __name__ == '__main__':
    if len(sys.argv) > 1:
        lang = sys.argv[1]
    else:
        lang = "go"
    dir = 'go/test'
    type_file = 'test'
    datasets = list()
    for i in range(2):
        datasets.append(lang + "_" + type_file + "_" + str(i) + ".jsonl")

    # for data in tqdm(datasets):

    # source_path = os.path.join(root_path,"mini_data/python/train.txt")
    codes_path = os.path.join(root_path, dir, "src_code_untoken.txt")
    title_path = os.path.join(root_path, dir, "tgt_title.txt")
    code_path = os.path.join(root_path, dir, "src_code.txt")

    with open(title_path, "w+") as title_file:
        with open(code_path, "w+") as code_file:
            with open(codes_path, "w+") as codes_file:
                for data in datasets:
                    source_path = os.path.join(root_path, "big_data", lang, type_file, data)
                    with open(source_path, "r", encoding="utf-8") as source_file:
                        for source in tqdm(source_file.readlines()):
                            code = find_code(source)
                            title = find_title(source)
                            codes = find_codes(source)
                            if not code or not title or not codes:
                                continue
                            try:
                                code_file.write(code + '\n')
                                codes_file.write(codes + '\n')
                                title_file.write(title + '\n')
                            except UnicodeEncodeError:
                                continue
                            except TypeError:
                                continue
                            # break
