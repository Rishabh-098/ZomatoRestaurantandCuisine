#!/usr/bin/env python
# coding: utf-8

# In[43]:


import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[2]:


df = pd.read_csv('Combined_data.csv')


# In[3]:


df.drop(columns=['Id','Urls','Restaurant_Name'],inplace = True)


# In[4]:


df


# In[5]:


# from sklearn.preprocessing import LabelEncoder
# lb = LabelEncoder()


# In[6]:


# obj_cols = [i for i in df.columns if df[i].dtypes == 'object']
# obj_cols


# In[7]:


# for i in obj_cols:
#     df[i] = lb.fit_transform(df[i])


# In[8]:


frq_loc = df.groupby("Location").size()


# In[9]:


frq_loc


# In[10]:


frq_dis_loc = df.groupby("Location").size()/len(df)


# In[54]:


frq_dis_loc


# In[51]:


z = pd.DataFrame(frq_dis_loc)


# In[53]:


z.to_csv('frq_dis_loc_map.csv')


# In[12]:


df_frq_loc = df.copy()


# In[13]:


df_frq_loc["loc_frq"] = df_frq_loc.Location.map(frq_dis_loc)


# In[14]:


df_frq_loc.drop("Location",axis = 1, inplace=True)


# In[15]:


df_frq_loc.head()


# In[16]:


frq_cus = df.groupby("Cuisines").size()


# In[17]:


frq_dis_cus = df.groupby("Cuisines").size()/len(df)


# In[55]:


frq_dis_cus


# In[56]:


f = pd.DataFrame(frq_dis_cus)


# In[57]:


f.to_csv('frq_dis_cus_map.csv')


# In[ ]:





# In[18]:


df_frq_cus = df_frq_loc.copy()


# In[19]:


df_frq_cus["loc_cus"] = df_frq_cus.Cuisines.map(frq_dis_cus)


# In[ ]:





# In[20]:


df_frq_cus.drop("Cuisines",axis = 1, inplace = True)


# In[21]:


df_frq_cus


# In[58]:


X = df_frq_cus.drop(columns=['Price_For_One','Latitude','Longitude','Rating','Delivery_review_number']).values
y = df_frq_cus['Price_For_One'].values


# In[59]:


X


# In[60]:


y


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state= 2,test_size= 0.2)


# In[63]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


# In[64]:


no_of_decision_tree = [10,20,30,40,50,60,70,80,90,100]
max_no_features  = ['sqrt','log2']
max_depth = [6,7,8,9,10,11,12,13,14,15]
criterian_for_decision_tree = ['gini','entropy']
min_sample_split = [1,2,3,4,5]


# In[65]:


random_grid ={
    'n_estimators':no_of_decision_tree,
    'max_features':max_no_features,
    'max_depth':max_depth,
    'criterion':criterian_for_decision_tree,
    'min_samples_split':min_sample_split
}


# In[66]:


from sklearn.model_selection import RandomizedSearchCV

rscv = RandomizedSearchCV(estimator=rfc,param_distributions=random_grid,n_iter=25,cv=5,n_jobs=-1)


# In[67]:


rscv.fit(X_train,y_train)


# In[68]:


rscv.best_params_


# In[69]:


rscv.best_estimator_


# In[70]:


rc_final= RandomForestClassifier(n_estimators = 100,min_samples_split = 3,max_features ='log2',max_depth= 15,criterion= 'entropy')


# In[71]:


rc_final.fit(X_train,y_train)


# In[72]:


y_pred = rc_final.predict(X_test)


# In[73]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix


# In[74]:


accuracy_score(y_test,y_pred)


# In[75]:


print(classification_report(y_test,y_pred))


# In[76]:


y_pred


# In[77]:


y_test


# In[78]:


filename = 'price_predictor'
pickle.dump(rc_final, open(filename,'wb'))


# In[79]:


load_model = pickle.load(open(filename,'rb'))
load_model.predict(X_test)


# In[ ]:




