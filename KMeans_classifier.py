# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:00:35 2020
首先爬取豆瓣上所有的书籍信息作为基本的语料文本，大概7000个文本数据 去重后1839篇书评
豆瓣读书数据聚类项目：
1.数据提取
2.数据归一化
3.k_means模型构建：文本数据矩阵化，构建模型、训练
csv格式文件怎么查看编码方式 txt打开另存为修改
utf_8_sig 还是 utf-8-sig
@author: 地三仙
"""

#import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 特征向量
def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    feature_type = feature_type.lower().strip()
 
    if feature_type == 'binary': # 这是什么模型？
        vectorizer = CountVectorizer(binary=True,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=10000)
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")
 
    feature_matrix = vectorizer.fit_transform(documents).astype(float)
 
    return vectorizer, feature_matrix



book_data_raw = pd.read_csv("./data/book_data.csv", encoding='utf-8')
print(book_data_raw.head().iloc[:, 3:4])
print(book_data_raw.columns)
book_data_raw['content'][:5]
book_data_raw.loc[book_data_raw['title'] == '月亮与六便士',].iloc[:, 0:2]
book_data = book_data_raw.drop_duplicates(subset=['title'], keep='first').copy()  # 避免后面赋值报错
book_data_uc = book_data_raw.drop_duplicates(subset=['title'], keep='first')  # 避免后面赋值报错
print(book_data_uc.index.tolist()[-20: ])
print(book_data.index.tolist()[-20: ])
print(book_data_raw.index.tolist()[-20: ])

#book_data.loc[book_data['title'] == '小王子',].iloc[:, 0:2]
#contents = book_data.loc[book_data['title'].isin(['追风筝的人','霍乱时期的爱情','怪诞故事集']),'content'].tolist()
#titles = ['追风筝的人','霍乱时期的爱情','怪诞故事集']
#for title, con in zip(titles,contents):
#    print(title + ":",con)
# 提取文本数据
book_titles = book_data['title'].tolist()
book_content = book_data['content'].tolist()
print(len(book_content))
# 数据归一化
from normalization import normalize_corpus
corpus = normalize_corpus(book_content)  # 还有特殊符号 【 】 ★  ─ ─ ◆    × 嫌疑人x 这个可以保留
# K-means模型构建
# 提取tf-idf特征
# min_df、max_df词的频率筛选
#如果ngram_range = (1, 3) 表示选取1到3个词做为组合方式: 词向量组合为: 
#    'I', 'like', 'you', 'I like', 'like you', 'I like you' 构成词频标签
# ngram_range=(1, 2) 表示1到2个词作为组合方式 这里不起作用 tfidf并没有这个参数
# 对于中文词汇来说好像不太有必要
vectorizer, feature_matrix = build_feature_matrix(corpus, feature_type='tfidf',
                                                  ngram_range=(1, 2), min_df=0.2, max_df=0.8)  


# 查看特征数量

print(feature_matrix.shape)
# 获取特征名字  14944 这是不是太多了？  可以通过max_features控制
feature_names = vectorizer.get_feature_names()                                          
len(feature_names)
print(vectorizer.vocabulary_)  
print(feature_matrix.todense().shape)

# 开始聚类 得到12个类别
from sklearn.cluster import KMeans
def k_means(feature_matrix, num_clusters=12):
    km = KMeans(n_clusters=num_clusters, max_iter=1000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters
num_clusters = 12
km_obj, clusters = k_means(feature_matrix, num_clusters)
book_data['cluster'] = clusters
cluster_0 = book_data.loc[book_data['cluster'] == 0,:]['title'].tolist()
print(len(cluster_0))
# 打印聚类信息
len(km_obj.cluster_centers_[0])
len(km_obj.cluster_centers_)

def get_cluster_data(clustering_obj, book_data, feature_names, 
                     num_clusters,topn_features=12):
    cluster_details = {}
    # 获取cluster的center
# argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
# [::-1] 倒序输出 为-2时 按索引-1,,-3,-5 这样输出
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # 获取簇的关键特征
    # 获取簇的书名
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index] for index in 
                        ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features
        books = book_data.loc[book_data['cluster'] == cluster_num, :]['title'].tolist()  # .values.tolist() 更标准？
        cluster_details[cluster_num]['books'] = books
    return cluster_details
       
cluster_data =  get_cluster_data(clustering_obj=km_obj, book_data=book_data, 
                                 feature_names=feature_names, num_clusters=num_clusters,
                                 topn_features=6)

# 打印
for cluster_num, cluster_details  in cluster_data.items():
    print("cluster %d 详细信息：" % cluster_num)
    print("--" * 20)
    print("key_fratures:{}".format(cluster_details['key_features']))
    print("books in this cluster:")
    books = cluster_details.get('books', [])
    if len(books) > 0:
        print(",".join(books))
    