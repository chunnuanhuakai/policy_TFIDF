# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:17:16 2018

@author: zbj
"""

root = "E:\\liuhongbing\\work\\policy_tong_0712\\data_v1\\technology-services-policy\\model_train\\"
tc = root + 'traincorpus2.xlsx'
import pandas as pd

data = pd.ExcelFile(tc)
df = pd.read_excel(data, 'sheet1')

dataFilter = df[df['人工打标'].isnull()][['模型打标的政策','title','cont']]

outresult = "E:\\liuhongbing\\work\\policy_tong_0712\\data_v1\\technology-services-policy\\model_train\\traincorpus2"
dataFilter.to_csv(outresult, sep='\t', header=False, index=False)


def checkbgcz(query):
    if '兼并' in query:
        return True
    if '收购' in query:
        return True
    if '重组' in query:
        return True
    if '关闭企业' in query:
        return True
    return False



def data_norm(data):
    
    #discard the strange sign 
    pattern = "[A-Za-z0-9<>&#\n\t\-\:\=\/\_\"\!\%\[\]\;\?\宋体\仿宋\微软\(\)\()\（）]"
    data['cont'] = data['cont'].apply(lambda x : re.sub(pattern, "", x))
    
    #discard the space
    data['cont'] = data['cont'].apply(lambda x : x.replace(' ', ''))
    
    #discard the \n
    data['cont'] = data['cont'].apply(lambda x : x.replace('\r', ''))
    
    #fetch the before 130 word
    data['cont'] = data['cont'].apply(lambda x : x[0:130])
    
    # titcont == title + cont
    data['titcont'] = data['title']+data['cont']
    
    return data
    

def titlecont(row):
    if row['cont'] is np.nan:
        return row['title']
    else:
        return row['title']+row['cont']
    
    
    
    
def data_tran(path, outpath,policy):
    xlsx = pd.ExcelFile(path)
    data = pd.read_excel(xlsx, 'Sheet1')
    data = data[['title','content']]
    data.drop_duplicates(subset=['content'],inplace=True)
    data.rename(columns={'content':'cont'},inplace=True)
    data_norm(data)
    data['label'] = policy
    data = data[['label','titcont']]
    data.to_csv(outpath, sep='\t',header=False,index=False)
    
    
    
    
    
    
    
    
    