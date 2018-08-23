# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:37:23 2018

@author: zbj
"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import pickle
import numpy as np

class policy_model_train:
    
    
    def __init__(self, path):
        self.path = path
        
    def writebunchobj(path, bunchobj):
        with open(path, 'wb') as file_obj:
            pickle.dump(bunchobj, file_obj)
        
            
    def testsampletran(self):
        test_sample = {}
        x_test_tmp = []
        y_test_tmp = []
        for index in range(len(self.x_test)):
            if self.x_test.iloc[index] in test_sample.keys():
                tmp = test_sample[self.x_test.iloc[index]]
                tmp.append(self.y_test.iloc[index])
                test_sample[self.x_test.iloc[index]] = tmp
            else:
                tmp = []
                tmp.append(self.y_test.iloc[index])
                test_sample[self.x_test.iloc[index]] = tmp
                
        for key,value in test_sample.items():
            x_test_tmp.append(key)
            y_test_tmp.append(value)
        
        self.x_test_filter = x_test_tmp
        self.y_test_filter =y_test_tmp
        
        
    def tfvectorize(self):
        
        # divide the trainSet and testSet
        data = pd.read_table(self.path + '\\corpus_seg', sep='\t',header=None)
        data.rename(columns={0:'label',1:'title'},inplace=True)
        y = data['label']
        x = data['title']
        self.x_train, self.x_test,self.y_train,self.y_test = train_test_split(x, y, test_size=0.1,random_state=0)
        print('x_train length:',len(self.x_train),'\t', len(self.y_train))
        print('y_test length:',len(self.x_test),'\t', len(self.y_test))
        
        self.testsampletran()
        ## covert corpus into word of bag
        cv = CountVectorizer(max_features = 3000)
               # get word of bag
        X = cv.fit_transform(self.x_train)
        self.x_train_tf = X
        self.x_test_tf = cv.transform(self.x_test_filter)

        # get word 
        self.word=cv.get_feature_names()

        
        ## calc the tfidfvectorizer
        tv = TfidfTransformer()
        self.x_train_tfidf = tv.fit_transform(self.x_train_tf)
        self.x_test_tfidf = tv.transform(self.x_test_tf)
        print(X.toarray())
        
        
    # 朴素贝叶斯分类
    def NaiveBaysTrain(self):
        clf = MultinomialNB(alpha=0.01)
        clf.fit(self.x_train_tfidf,self.y_train)
        self.classes_ = clf.classes_
        
        self.pre = clf.predict(self.x_test_tfidf)
        self.pre_pro = clf.predict_proba(self.x_test_tfidf)
        
    # LR 回归分类
    def LogisticRegressionTrain(self):
        cls = LogisticRegression(multi_class='multinomial',solver='lbfgs')
        cls.fit(self.x_train_tfidf,self.y_train)
        self.classes2_ = cls.classes_
        
        self.pre_prob_lr = cls.predict_proba(self.x_test_tfidf)
        
       
    def calc_score(self, pre_prob, classes_):
        
        count = 0
        for i in range(len(pre_prob)):
            tmp = pre_prob[i]
            tmpsortindex = tmp.argsort()
            
            v1index = tmpsortindex[19] # 序列
            v1 = tmp[v1index]  # 得分
            pol1 = classes_[v1index] # 政策类别
            v2index = tmpsortindex[18]
            v2 = tmp[v2index]
            pol2 = classes_[v2index] # 政策类别
            # 取两个结果
            if v1-v2<0.1:
                if len(self.y_test_filter[i])==1:
                    continue
                else:
                    if (pol1 in self.y_test_filter[i]) & (pol2 in self.y_test_filter[i]):
                        count = count+1
            else :
                if len(self.y_test_filter[i])>=2:
                    continue
                else:
                    if pol1 in self.y_test_filter[i]:
                        count = count +1
        
        print('召回率：', 1.0* count/len(pre_prob))
        
    def train_NB(self):
        self.tfvectorize()
        self.NaiveBaysTrain()
        self.LogisticRegressionTrain()
        self.calc_score(self.pre_pro, self.classes_)
        self.calc_score(self.pre_prob_lr, self.classes2_)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    