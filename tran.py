from translate import Translator
import os


root_path = os.path.dirname(__file__)
path = os.path.join(root_path,"result.txt")
translator= Translator(to_lang="chinese")

with open(path, "r", encoding="utf-8") as file:
        for input in file.readlines():
            translation = translator.translate(input)

with open(path, "a+", encoding="utf-8") as file:
    file.write(translation)
    