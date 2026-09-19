import numpy as np                  # 数学计算包
import pandas as pd                 # 数据处理包
import matplotlib.pyplot as plt     # 画图包
import jieba                        # 分词包
from sklearn.feature_extraction.text import CountVectorizer # 词频统计包, 把评论内容 转成 词频矩阵.
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB               # 朴素贝叶斯对象

df = pd.read_csv('data/书籍评价.csv',encoding='gbk')
df['lables'] = np.where(df['评价']=='好评',1,0)
y = df['lables']

#分词获得x
comment_list = [','.join(jieba.lcut(i)) for i in df['内容']]
# print(comment_list)

#加载停用词列表
with open('data/stopwords.txt','r',encoding='utf-8') as f:
    stopwords_list = f.readlines()
    stopwords_list = [i.strip() for i in stopwords_list]
    stopwords_list = list(set(stopwords_list))

# print(stopwords_list)

transfer = CountVectorizer(stop_words=stopwords_list)
x = transfer.fit_transform(comment_list).toarray()

x_train = x[:10]
y_train = y[:10]
x_test = x[10:]
y_test = y[10:]

model = MultinomialNB()
model.fit(x_train,y_train)
y_pre = model.predict(x_test)
print(accuracy_score(y_test,y_pre))