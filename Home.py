import streamlit as st

st.set_page_config(
    page_title = "Zomato Restraunt Selector", layout= "wide"
)

st.markdown("<h1 style='text-align: center; color: white;'> Zomato Restraunt Advisor </h1>", unsafe_allow_html=True)

#---------Introduction section ---------
with st.container():
    st.subheader("Introduction of the Project")
    st.write("This project in collaboration with masai as a Machine Learning project. The Objective of this project is to create a machine learning model and UI to suggest people different restaraunt based on their budget and preffered location. ")
#-----------Image of zomato-----------------
image = open('Screenshot 2023-02-18 160304.png', "rb").read()
st.image(image, use_column_width=True, 
             width=None, clamp=False, channels="RGB", output_format="auto")

#-----------Steps Taken-------------------
with st.container():
    st.subheader("Steps Taken in this project:")
    
    #defining display fu
    def display_step(step_number, step_text):
        st.write(f"Step {step_number}: {step_text}")
    steps = [
        ("1", "Extracting the data"),
        ("2", "sanitization of data"),
        ("3", "Analysis of data"),
        ("4", "Building a Machine learning model"),
        ("5", "Creating Front-End to display all the information")
        ]
    
    for step_number, step_text in steps:
        display_step(step_number, step_text)
    