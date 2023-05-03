import streamlit as st
from streamlit_lottie import st_lottie
import json
st.set_page_config(
    page_title = "Case Study", layout= "wide"
)


st.header("You are hired in a consultancy firm , one of their client want to open a remote kitchen (Only delivery) in Bangalore suggest them which location will be suitable for their restaurant and what should be the price of different types of dishes in early days. ")


def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

lottielink = load_lottiefile("data.json")

st_lottie(
    lottielink, 
    speed = 1,
    reverse=False,
    loop=True,
    quality="high",
    height = None,
    width = None, 
    key = None 
)



with st.container():
    st.write(
        """
        After analyzing the data on the area-wise distribution of restaurants in Bangalore, it is recommended that the client open their remote kitchen in the BTM area. 
        This is because this location has the highest number of restaurants with delivery reviews greater than 1000, which indicates a high demand for delivery services in the area.

In terms of pricing, it is recommended that the client set the price range for their dishes between 50 to 350 rupees. 
This will ensure that the prices are competitive and affordable for customers.

Furthermore, based on the data on the average price for different types of cuisines in the BTM area, it is suggested that the client set the average price for their dishes around 170 rupees, which is the average price for most cuisines in this area. This will enable the client to offer competitive pricing while also ensuring profitability.

In the initial stages of the restaurant, it is important for the client to focus on building a customer base and gaining traction. 
They can do this by offering promotional discounts or freebies to attract customers and build brand awareness.
The client can also gather customer feedback to improve the quality of their dishes and service. 
As the business grows, the client can adjust their pricing strategy based on the market demand and competition.
   """ )
    
    
