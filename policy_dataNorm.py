# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:57:51 2018

@author: zbj
"""

from sklearn import svm
import jieba
import numpy as np
import pandas as pd
import re

class data_transform:
    
    train_data = []
    train_label=[]
    
    stop_words = []
    path = ""
    
    def __init__(self,path):
        self.path = path
        
    
    def getstopword(self):
        self.stop_words = [line.strip() for line in open(self.path+'\\stopWords', 'r', encoding='utf-8').readlines()]    
       

    def gettrain_data(self):

        with open(self.path+"\\trainCorpus.txt",'r',encoding='utf-8') as f:
            for line in f.readlines():
                lineseg = line.split('\t')
                if len(lineseg)>=2 and len(lineseg[1])>1:
                    data_temp = ""
                    titlefilter = re.sub(r'([\d]+)','',lineseg[1])
                    
                    for word in jieba.cut(titlefilter, cut_all = False):
                        if word not in self.stop_words:
                            data_temp = data_temp+'/'+word
                    self.train_data.append(data_temp)
                    
                    # add the sample label
                    if len(data_temp)>=1:
                        self.train_label.append(lineseg[0])
    
    def getwritefile(self):
        outpath = self.path + '\\corpus_seg'
        fwrite = open(outpath, 'w',encoding='utf-8')
        
        for i in range(len(self.train_label)):
            fwrite.write(self.train_label[i]+"\t"+self.train_data[i])
            
        fwrite.flush()
        fwrite.close()
        
        
    def data_train(self):
        self.getstopword()
        self.gettrain_data()
        self.getwritefile()
    