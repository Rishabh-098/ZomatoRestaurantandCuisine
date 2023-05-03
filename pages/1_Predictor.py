import streamlit as st 
import pandas as pd 
import pickle
import os
import numpy as np

filename = 'Machine_learning_model\price_predictor'
load_model = pickle.load(open(filename,'rb'))
file_name = 'Machine_learning_model\Cuisine_predictor'
cuisine_model = pickle.load(open(file_name,'rb'))

df = pd.read_csv('Dataset\Combined_data.csv')
loc_map = pd.read_csv('D:/webapp/Dataset/frq_dis_loc_map.csv')
cus_map = pd.read_csv('D:/webapp/Dataset/frq_dis_cus_map.csv')

st.title('Restaraunt Selector')

column_left, column_right = st.columns(2)
with column_left:
    filter_location = st.selectbox('Preferred Location:' , options= df['Location'].unique()) # Dropdown Menu For Different Locations

    best_cuisine = df[df['Location'] == filter_location].groupby('Cuisines')['Delivery_review_number'].sum().idxmax() # Finding best Cuisine for selected Location

    st.write(f"For the Selected area, The Most Popular Cuisine is: **{best_cuisine}**.")


#-------------------calculating average price for 1 person for selected location
    filter_location_result = df[df['Location'] == filter_location]
    avg_price = filter_location_result['Price_For_One'].mean()
    st.write(f"And the average Price for One Person is: **₹{avg_price:.2f}**. ")

#----------------------for selected location, filtering the best Restaurant and list of cuisine they serve. 

    highest_rating = filter_location_result['Delivery_review_number'].max()
    best_restaurant = df[(df['Location'] == filter_location) & (df['Delivery_review_number'] == highest_rating)]
    cuisine_list = best_restaurant['Cuisines'].tolist()
    restaurant_name = best_restaurant['Restaurant_Name'].unique().tolist()
    

with column_right:
    filter_cuisine = st.selectbox('Preferred Cuisine:', options= df['Cuisines'].unique())


    highest_rating_cuisine = df[df['Cuisines'] == filter_cuisine]['Delivery_review_number'].max()
    best_restaurant_cuisine = df[(df['Cuisines'] == filter_cuisine) & (df['Delivery_review_number'] == highest_rating_cuisine)]
    #st.write(f"The best restaurant with highest Rating is {best_restaurant_cuisine['Restaurant_Name'].iloc[0]} for Your Preferred Cuisine in {best_restaurant_cuisine['Location'].iloc[0]}")
    st.write(f"For the selected Location Most Popular Restaurant is: **{restaurant_name[0]}** and Cuisines that they serve are: **{cuisine_list}**  ")
    st.write(f"You can Visit {best_restaurant_cuisine['Restaurant_Name'].iloc[0]} in {best_restaurant_cuisine['Location'].iloc[0]} as per Your Preferred Cuisine. ")
filter_price =  st.slider('Preferred price for one:', min_value=50,max_value=500,value = 50, step = 50)




#mapping location input to our encoding
user_input_location = filter_location
matching_row = loc_map.loc[loc_map['Location'] == user_input_location]
index = matching_row.index[0]
value_location = loc_map.iloc[index, 1]
#mapping location input to our encoding
user_input_cuisine = filter_cuisine
matching_row = cus_map.loc[cus_map['Cuisines'] == user_input_cuisine]
index = matching_row.index[0]
value_cuisine = cus_map.iloc[index, 1]


def price_prediction(input_data):
    input_data_as_numpy = np.asarray(input_data)
    input_array_reshape = input_data_as_numpy.reshape(1,-1)
    predict_price = load_model.predict(input_array_reshape)
    return predict_price
    



columnleft,columnright = st.columns(2)
price = price_prediction([value_location,value_cuisine])
with columnleft:
    st.header("Recommended Price")

    st.write(f"Recommended Price according to your preference is: {price}")


def cuisine_prediction(input_data):
    input_data_as_numpy = np.asarray(input_data)
    input_array_reshape = input_data_as_numpy.reshape(1,-1)
    predict_cuisine = cuisine_model.predict(input_array_reshape)
    return predict_cuisine

cuisine = cuisine_prediction([filter_price,value_location])
with columnright:
    st.header("Recommended Cuisine")

    st.write(f"Recommended Cuisine according to your preference is: {cuisine}")



