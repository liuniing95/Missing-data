#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy import stats


# In[2]:


def getDt(name):
    url = '/Users/liuning/Desktop/data science/dt/Incomplete+Datasets/' + name + '.xlsx'
    excel = pd.read_excel(url,index_col=None, header = None)
    url1 = '/Users/liuning/Desktop/dt/' + name + '.csv'
    excel.to_csv(url1, encoding = 'UTF-8',index=False)
    df = pd.read_csv(url1, header = 0)
    return df


# In[3]:


def getOldDt(name):
    url = '/Users/liuning/Desktop/data science/dt/Original+Datasets/'+ name + '.xlsx'
    excel = pd.read_excel(url,index_col=None,header = None)
    url1 = '/Users/liuning/Desktop/dt/BCW/Complete.csv'
    excel.to_csv(url1, encoding = 'UTF-8',index=False)
    df = pd.read_csv(url1, header = 0)
    return df


# In[4]:


def preprocessing(missing_dt, complete_dt):
    sorted_index = list(missing_dt.isna().sum().sort_values().index)
    if missing_dt[sorted_index[0]].isna().values.any() == True:
        mean = np.mean(missing_dt[sorted_index[0]])
        missing_dt[sorted_index[0]] = missing_dt[sorted_index[0]].fillna(mean)
    old_dt = complete_dt.reindex(columns = sorted_index)
    new_dt = missing_dt.reindex(columns = sorted_index)
    return new_dt, old_dt


# In[5]:


def NRMS(imp_dataset, old_dataset):
    return ((imp_dataset.values - old_dataset.values)**2).sum() / ((old_dataset.values)**2).sum()


# In[6]:


def impute(dt):
    sorted_index = list(dt.isna().sum().sort_values().index)

    y_miss_list = []
    y_miss_row = []
    y_miss_columns = []

    for i in range(0,len(sorted_index)):
        if dt[sorted_index[i]].isna().values.any() == True:
            for j in range(0,len(dt.index)):
                if pd.isnull(dt.loc[j, sorted_index[i]]) == True:
                    y_miss_list.append(dt.loc[j, sorted_index[i]])
                    y_miss_row.append(j)
                    y_miss_columns.append(sorted_index[i])
        
    for i in range(0,len(y_miss_list)):
        full_index = dt.columns[dt.notnull().all()].tolist()
        y_miss = y_miss_list[i]
        x_miss = dt.loc[[y_miss_row[i]]].reindex(columns = full_index).values.reshape(1, -1)
        y_obs = dt[y_miss_columns[i]].dropna(how = 'all')
        x_obs = dt.reindex(columns = full_index, index = y_obs.index)
        clf = RandomForestClassifier(n_estimators = 15)
        clf.fit(x_obs,y_obs.astype('int'))
        dt.loc[y_miss_row[i], y_miss_columns[i]] = clf.predict(x_miss)
    return dt


# In[7]:


def result():
    nrms = 100
    while nrms > 1:
        df = impute(new_df)
        nrms = NRMS(df,old_df)
    return nrms


# In[8]:


new_df = getDt('BCW/planned 1')
old_df = getOldDt('BCW')

new_df, old_df = preprocessing(new_df, old_df)

#print(new_df.dtypes)
#new_df = impute(new_df)
#old_df
print(result())
#new_df


# In[ ]:





# In[ ]:




