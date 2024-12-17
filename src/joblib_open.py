'''
加载并使用模型
'''
from joblib import load # 加载模型
import pandas as pd # Pandas数据处理
from nltk.stem.porter import PorterStemmer # 词干提取器
import re # 正则表达式
from sklearn.metrics import accuracy_score # 准确率
import os # 操作系统相关

# 读取文件
def read_text():
    with open('input_text.txt', 'r' , encoding = "utf-8") as f:
        text = f.read()
    return text

text = read_text()

# PorterStemmer词干提取
def stemming(content):
    stemmer = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) # 使用正则表达式将所有非字母字符替换为空格
    stemmed_content = stemmed_content.lower() # 使用Python的lower()方法将所有字母转换为小写
    stemmed_content = stemmed_content.split() # 使用Python的split()方法将文本分割为单词
    stemmed_content = [stemmer.stem(word) for word in stemmed_content] # 使用PorterStemmer将单词提取词干
    stemmed_content = ' '.join(stemmed_content) # 使用Python的join()方法将提取词干后的单词重新连接为文本
    return stemmed_content

text = stemming(text)


# 加载并使用模型
def load_model(input):
    model_path = os.path.join('models', 'best_model.joblib')
    loaded_best_model = load(model_path)
    prediction = loaded_best_model.predict(pd.Series(input))
    with open("input_text.txt", "w" ,encoding = "utf-8") as f:
        f.write("")
    return prediction

prediction = load_model(text)