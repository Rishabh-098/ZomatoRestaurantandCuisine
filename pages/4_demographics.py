import streamlit as st
import plotly_express as px
import pandas as pd

st.set_page_config(page_title= "Dashboard", page_icon= ":bar_graph:", layout= "wide")
df = pd.read_csv('Dataset\Combined_data.csv')
st.markdown("<h1 style='text-align: center; color: white;'> Zomato Restraunt Dashboard </h1>", unsafe_allow_html=True)

####-----------------Different Filters------------------
st.sidebar.header("Please Filter Here:")
location = st.sidebar.selectbox(
    "Preferred Location:",
    options= df["Location"].unique()  
)


cuisines = st.sidebar.selectbox(
    "Preferred Cuisines:",
    options= df["Cuisines"].unique()  
)

#------------------Graph and Dashboard Component---------------------------
#changing dataset based on filter ------------------
df_selection = df.query(
    "Location == @location & Cuisines == @cuisines"
)
st.dataframe(df_selection)

#---------------------------------------------------------------
left_column,right_column = st.columns(2)
with left_column:
    st.header("Highest Rated Restraunt:")
    try:
        max_review_restaurant = df_selection.loc[df_selection['Delivery_review_number'].idxmax(), 'Restaurant_Name']
        st.subheader(max_review_restaurant)
    except ValueError:
        st.error("No Restraunt found for the provided filters")
with right_column:
    st.header("Cuisines with Highest Rating: ")
    try:
        max_cuisine_rating = df_selection.loc[df_selection['Delivery_review_number'].idxmax(), 'Cuisines']
        st.subheader(max_cuisine_rating)
    except:
        st.error("No Cuisines found for the provided filters")
        
#----------------------graphs for area wise distribution---------------------------------------
with st.container(): 
    st.markdown("<h1 style='text-align: center; color: white;'>Area-wise distribution of restaurant</h1>", unsafe_allow_html=True)
    restaurant_count = df.groupby('Location').size().reset_index(name='Count')
    fig_area_dis_bar = px.bar(restaurant_count, x = "Location", y = "Count", orientation = "v",color_discrete_sequence=["#0083B8"] * len("Count"))
    fig_area_dis_bar.update_layout(height = 750)
    st.plotly_chart(fig_area_dis_bar, use_container_width= True)
    
#-------------------------location with max number of restraunt and reviews > 1000--------------
with st.container():
    left_column1,right_column1 = st.columns(2)
    with left_column1:
        st.subheader("Based on data provided location with highest number of Restraunts and Reviews greater than 1000 are:")
        x = st.text_input(label= "Enter Number of Restraunt required")
        filtered_df = df[df['Delivery_review_number'] > 1000]
        grouped_df = filtered_df.groupby('Location')['Id'].nunique()
        sorted_df = grouped_df.sort_values(ascending=False)  
        try:
            st.write(sorted_df.head(int(x)))
        except ValueError:
            st.error("Enter a valid number")
    with right_column1:
        st.subheader("Location with maximum number of selected rated restaurant.")
        threshold = st.slider("Select the review threshold", 0, 13500, 100)
        filtered_df = df[df['Delivery_review_number'] < threshold]
        grouped_df = filtered_df.groupby('Location')['Id'].nunique()
        sorted_df = grouped_df.sort_values(ascending=False)
        top_locations = sorted_df.head(5)
        st.write(f"The top 5 locations with the maximum number of restaurants with less than {threshold} reviews are:")
        st.write(top_locations)

#------------------Number of restaurant for each type of cuisine.-------------------
with st.container():
    selected_cuisine = st.selectbox("Select a cuisine", df['Cuisines'].unique())
   
    filtered_df = df[df['Cuisines'] == selected_cuisine]
    restaurant_count = len(filtered_df)
    st.write(f"The number of restaurants for {selected_cuisine} cuisine is: {restaurant_count}")    
    
    grouped_df = filtered_df.groupby('Location')
    avg_price_df = grouped_df['Price_For_One'].mean()
    cheap_location = avg_price_df.idxmin()
    expensive_location = avg_price_df.idxmax()
    cheap_df = filtered_df[filtered_df['Location'] == cheap_location]
    expensive_df = filtered_df[filtered_df['Location'] == expensive_location]
    st.write(f"The cheap {selected_cuisine} restaurants in {cheap_location} are:")
    st.write(cheap_df[cheap_df['Price_For_One'] == cheap_df['Price_For_One'].min()])    
    st.write(f"The expensive {selected_cuisine} restaurants in {expensive_location} are:")
    st.write(expensive_df[expensive_df['Price_For_One'] == expensive_df['Price_For_One'].max()])
    st.write(f"Average price of one {selected_cuisine} meal by location:")
    st.write(avg_price_df)