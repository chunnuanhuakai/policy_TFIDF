# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:59:55 2018

@author: zbj
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import pickle
import numpy as np
import operator

class policy_model_train:
    
    
    def __init__(self, path):
        self.path = path
        
    
        
#     #利用训练样本选 计算词向量  
#    def sample_2_w2v_2(self):
#        
#        policy_model_w2v = pd.read_table(self.path + '\\policy_word2vec', sep=' ',header=None)
#        policy_model_w2v.set_index(0, inplace=True)
#        self.dict_w2v = list(policy_model_w2v.index)
#        
#        
#        # divide the trainSet and testSet
#        data = pd.read_table(self.path + '\\corpus_seg_w2v', sep='\t',header=None)
#        data.rename(columns={0:'label',1:'title'},inplace=True)
#        y = data['label']
#        x = data['title']
#        x_train, self.x_test,self.y_train,self.y_test = train_test_split(x, y, test_size=0.2,random_state=0)
#        print('x_train length:',len(x_train),'\t', len(self.y_train))
#        print('y_test length:',len(self.x_test),'\t', len(self.y_test))
#        self.x_train_w2v = []
#        self.x_test_w2v = []
#        
#        vec_dim = policy_model_w2v.shape[1]
#        
#        sample_nums_train = 0
#        # covert train sample into  vector
#        for train_sample in x_train:
#            vector = np.zeros(vec_dim)
#            count = 0
#            for word in train_sample.strip().split('/'):
#                if word in self.dict_w2v:
#                    
#                    vector += policy_model_w2v.ix[word]
#                    count += 1
#                
#                    
#            if count >0:
#                vector = vector / count
#            self.x_train_w2v.append(vector)
#            
#            sample_nums_train += 1
#            if sample_nums_train % 10 == 0:
#                print("train_nums:",sample_nums_train,"\t","rate:", sample_nums_train/len(x_train))
#        
#        self.testsampletran()
#                
#        # covert test sample into vec
#        sample_nums = 0
#        for test_sample in self.x_test_filter:
#            
#            vector = np.zeros(vec_dim)
#            count = 0
#            for word in test_sample.split('/'):
#                if word in self.dict_w2v:
#                    vector += policy_model_w2v.ix[word]
#                    count += 1
#                
#            if count >0:
#                vector = vector / count
#            self.x_test_w2v.append(vector)
#            
#            sample_nums += 1
#            if sample_nums % 10 == 0:
#                print("test_nums:",sample_nums,"\t","rate:", sample_nums/len(self.x_test_filter))
      

    # 训练语料 转为   w2v向量  
#    def sample_2_w2v(self):
#        
#        policy_model_w2v = pd.read_table(self.path + '\\policy_word2vec', sep=' ',header=None)
#        policy_model_w2v.set_index(0, inplace=True)
#        self.dict_w2v = list(policy_model_w2v.index)
#        vec_dim = policy_model_w2v.shape[1]
#        sample_x_w2v = []
#        sample_label = []
#        
#        sample_nums = 0
#        for line in open(self.path + "\\corpus_seg", 'r', encoding='utf-8').readlines():
#            seg = line.split("\t")
#            if len(seg)>=2:
#                vector = np.zeros(vec_dim)
#                count = 0
#                for word in seg[1].split('/'):
#                    if word in self.dict_w2v:
#                        vector += policy_model_w2v.ix[word]
#                        count += 1
#                if count>0:
#                    vector = vector / count
#                sample_x_w2v.append(list(vector))
#                sample_label.append(seg[0])
#                
#            sample_nums += 1
#            if sample_nums % 10 == 0:
#                print("nums:",sample_nums)
#                
#        outpath = self.path + '\\corpus_seg_w2v'
#        fwrite = open(outpath, 'w',encoding='utf-8')
#        
#        for i in range(len(sample_label)):
#            print('iter:----------------->',i)
#            fwrite.write(sample_label[i]+"\t"+str(sample_x_w2v[i]))
#            fwrite.write('\n')
#            
#        fwrite.flush()
#        fwrite.close()



    def check_label_sample(str1, str2):
        if str1 == str2:
            return str1
        else:
            return str1+'/'+str2
        
    ## 测试样本 标准化
    def test_sample_norm(self,x_test, y_test):
        
        x_test_str = []
        for index in range(len(y_test)):
            x_test_str.append("||".join("%s" % s for s in x_test[index]))
        
        test_data = pd.DataFrame({"label":y_test, "sample":x_test_str})
        data_drop_f = test_data.drop_duplicates(subset=['sample'], keep='first')
        data_drop_l = test_data.drop_duplicates(subset=['sample'], keep='last')
        data_drop_l.rename(columns={'label': 'label_2'},inplace=True)
        
        print(len(data_drop_f['label']), len(test_data['label']))
        if len(data_drop_f['label'])==len(test_data['label']):
            self.x_test = x_test
            self.y_test = [[i] for i in y_test]
            return 
        else :
            data = pd.merge(data_drop_f, data_drop_l, how='left', on='sample')
            
        data['label_sum'] = data.apply(lambda row : check_label_sample(row['label'], row['label_2']),axis=1)
        
        data['label_v'] = data['label_sum'].apply(lambda x : x.split('/'))
        data['sample_w2v'] = data['sample'].apply(lambda x : [float(i) for i in x.split('||')])
        
        self.x_test = np.array(list(data['sample_w2v']))
        self.y_test = list(data['label_v']) 



    def data_split_train_test(self):
        
        #divide the trainSet and testSet
        data = pd.read_table(self.path + '\\corpus_seg_w2v', sep='\t',header=None)
        data.rename(columns={0:'label',1:'title'},inplace=True)
        y = data['label']
        data['w2v']= data['title'].apply(lambda x: list( float(i) for i in x.split(',')))
        x = np.array(list(data['w2v']))
        y= list(data['label'])
        

        self.x_train, x_test,self.y_train,y_test = train_test_split(x, y, test_size=0.1,random_state=0)
        print('x_train length:',len(self.x_train),'\t', len(self.y_train))
        print('y_test length:',len(x_test),'\t', len(y_test))
        
        self.test_sample_norm(x_test, y_test)
                          
        
    # svm 模型训练，输出每类的概率值    
    def SVMTrain(self):
        cls = SVC(C=1,  kernel='poly',decision_function_shape='ovo',probability = True)
        cls.fit(self.x_train, self.y_train)
        
        self.classes3_ = cls.classes_
        self.pre_pro_svm = cls.predict_proba(self.x_test)

        
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
                if len(self.y_test[i])==1:
                    continue
                else:
                    if (pol1 in self.y_test[i]) & (pol2 in self.y_test[i]):
                        count = count+1
            else :
                if len(self.y_test[i])>=2:
                    continue
                else:
                    if pol1 in self.y_test[i]:
                        count = count +1
        
        print('召回率：', 1.0 * count/len(pre_prob))  

        

        
        
    def train_SVM_w2v(self):
        self.data_split_train_test()
        self.SVMTrain()
        self.calc_score(self.pre_pro_svm, self.classes3_)
   
        
if __name__=='__main__':
        
    root = 'E:\\liuhongbing\\work\\policy_tong_0712\\data_v1\\python_data\\modelResource\\origin_corpus'
    pmt = policy_model_train(root)
    pmt.train_SVM_w2v()
        
        
        
        
        
        
        
        
        