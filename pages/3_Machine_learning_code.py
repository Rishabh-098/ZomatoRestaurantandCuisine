import streamlit as st

    # Add a title
st.set_page_config(
    page_title = "Code", layout= "wide"
)
st.title("Machine Learning code to predict Cuisine based on Location and Price for One")

    # Define a code snippet
code = """
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('Combined_data.csv')

df.drop(columns=['Id','Urls','Restaurant_Name'],inplace = True)
frq_loc = df.groupby("Location").size()
frq_dis_loc = df.groupby("Location").size()/len(df)
df_frq_loc = df.copy()

df_frq_loc["loc_frq"] = df_frq_loc.Location.map(frq_dis_loc)
df_frq_loc.drop("Location",axis = 1, inplace=True)
df_frq_loc.head()

X = df_frq_loc.drop(columns=['Cuisines','Latitude','Longitude','Rating','Delivery_review_number']).values
y = df_frq_loc['Cuisines'].values

from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state= 2,test_size= 0.2)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

no_of_decision_tree = [10,20,30,40,50,60,70,80,90,100]
max_no_features  = ['sqrt','log2']
max_depth = [6,7,8,9,10,11,12,13,14,15]
criterian_for_decision_tree = ['gini','entropy']
min_sample_split = [1,2,3,4,5]


random_grid ={
    'n_estimators':no_of_decision_tree,
    'max_features':max_no_features,
    'max_depth':max_depth,
    'criterion':criterian_for_decision_tree,
    'min_samples_split':min_sample_split
}

from sklearn.model_selection import RandomizedSearchCV

rscv = RandomizedSearchCV(estimator=rfc,param_distributions=random_grid,n_iter=25,cv=5,n_jobs=-1)

rscv.fit(X_train,y_train)

rscv.best_params_

rscv.best_estimator_

rc_final= RandomForestClassifier(n_estimators = 80,min_samples_split = 3,max_features ='sqrt',max_depth= 6,criterion= 'gini')

rc_final.fit(X_train,y_train)

y_pred = rc_final.predict(X_test)


from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix

accuracy_score(y_test,y_pred)


print(classification_report(y_test,y_pred))

filename = 'Cuisine_predictor'
pickle.dump(rc_final, open(filename,'wb'))

load_model = pickle.load(open(filename,'rb'))
load_model.predict(X_test)
    """

    # Display the code snippet
st.code(code, language="python")
    
    
    
st.title("Machine Learning code to predict ***Price For One*** based on Location and Cuisines")

    # Define a code snippet
code = """
import matplotlib.pyplot as plt
import pandas as pd
import pickle


df = pd.read_csv('Combined_data.csv')



df.drop(columns=['Id','Urls','Restaurant_Name'],inplace = True)

# from sklearn.preprocessing import LabelEncoder
# lb = LabelEncoder()


# obj_cols = [i for i in df.columns if df[i].dtypes == 'object']
# obj_cols


# for i in obj_cols:
#     df[i] = lb.fit_transform(df[i])



frq_loc = df.groupby("Location").size()


frq_dis_loc = df.groupby("Location").size()/len(df)

frq_dis_loc
z = pd.DataFrame(frq_dis_loc)

z.to_csv('frq_dis_loc_map.csv')

df_frq_loc = df.copy()


df_frq_loc["loc_frq"] = df_frq_loc.Location.map(frq_dis_loc)

df_frq_loc.drop("Location",axis = 1, inplace=True)

df_frq_loc.head()

frq_cus = df.groupby("Cuisines").size()

frq_dis_cus = df.groupby("Cuisines").size()/len(df)

frq_dis_cus

f = pd.DataFrame(frq_dis_cus)

f.to_csv('frq_dis_cus_map.csv')


df_frq_cus = df_frq_loc.copy()

df_frq_cus["loc_cus"] = df_frq_cus.Cuisines.map(frq_dis_cus)


df_frq_cus.drop("Cuisines",axis = 1, inplace = True)



df_frq_cus

X = df_frq_cus.drop(columns=['Price_For_One','Latitude','Longitude','Rating','Delivery_review_number']).values
y = df_frq_cus['Price_For_One'].values


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state= 2,test_size= 0.2)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

no_of_decision_tree = [10,20,30,40,50,60,70,80,90,100]
max_no_features  = ['sqrt','log2']
max_depth = [6,7,8,9,10,11,12,13,14,15]
criterian_for_decision_tree = ['gini','entropy']
min_sample_split = [1,2,3,4,5]

random_grid ={
    'n_estimators':no_of_decision_tree,
    'max_features':max_no_features,
    'max_depth':max_depth,
    'criterion':criterian_for_decision_tree,
    'min_samples_split':min_sample_split
}

from sklearn.model_selection import RandomizedSearchCV

rscv = RandomizedSearchCV(estimator=rfc,param_distributions=random_grid,n_iter=25,cv=5,n_jobs=-1)

rscv.fit(X_train,y_train)

rscv.best_params_

rscv.best_estimator_


rc_final= RandomForestClassifier(n_estimators = 100,min_samples_split = 3,max_features ='log2',max_depth= 15,criterion= 'entropy')

rc_final.fit(X_train,y_train)

y_pred = rc_final.predict(X_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix

accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))


filename = 'price_predictor'
pickle.dump(rc_final, open(filename,'wb'))

load_model = pickle.load(open(filename,'rb'))
load_model.predict(X_test)
    """

    # Display the code snippet
st.code(code, language="python")