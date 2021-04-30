# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:47:25 2020
文本数据规整化的预处理
@author: 地三仙
"""
#1.分词
#2.去掉停用词和特殊字符。 停用词是个性化的地方  而且特殊字符一定都要删除吗？
#3.特征变量生成

import re
import jieba
import string

with open("./data/stop_words.gb18030.txt", encoding='gb18030') as f:
    stopword_list = [line.strip() for line in f.readlines()]

#print(stopword_list)

def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens

    
def remove_special(text):
    tokens = tokenize_text(text)
#    for token in tokens:
#        if ',' in token:
#            print(',')
#    string.punctuation 这东西不包含顿号
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filter_tokens = [pattern.sub('', token) for token in tokens]
    filter_text = ' '.join(filter_tokens)
    return filter_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filter_tokens = [token for token in tokens if token not in stopword_list]
    filter_text = ' '.join(filter_tokens)
    return filter_text
    

# 语料规划化处理
def normalize_corpus(corpus, tokenize=False):
    normalize_corpus = []
    for text in corpus:
        text = remove_special(text)
        text = remove_stopwords(text)
        if tokenize:
            text = text.split(' ')
        normalize_corpus.append(text)
    
    return normalize_corpus
    


if __name__ == '__main__':
    text = '增值税发票及海关代征增值税专用缴款书及其它服务行业发票, 公路、内河运输发票'
    text = remove_special(text)
    text_removed_stopwords = remove_stopwords(text)
    corpus = ['增值税发票及海关代征增值税专用缴款书及其它服务行业发票, 公路、内河运输发票',
        '公司是<<深圳市海天实业有限公司>>,成立于深圳多年,有良好的贸易信誉. 长期为各大公司代开电脑发票和']
    nor_cor = normalize_corpus(corpus, False)
    
    print(nor_cor)
    
    
    