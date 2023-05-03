import streamlit as st

# Define the Streamlit app
def main():
    # Add a title
    
    st.set_page_config(
    page_title = "Code", layout= "wide"
)
    st.title("Code for Extracting Basic Information of all the Restaraunt in Bangalore")

    # Define a code snippet
    code = """
    from selenium import webdriver
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin
import time

driver = webdriver.Chrome("chromedriver.exe")
driver.get("https://www.zomato.com/bangalore")
time.sleep(2)

scroll_pause = 2
screen_height = driver.execute_script("return window.screen.height;")
i = 1
V = 0
with open('Table1.csv', 'a+', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    header = ['Name', 'Price_For_One', 'Cuisines', 'Rating', 'Urls']
    writer.writerow(header)


while True:
    driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))
    i = i + 1
    time.sleep(scroll_pause)
    scroll_height = driver.execute_script("return document.body.scrollHeight;")
    if i == 200:
        break

src = driver.page_source
soup = BeautifulSoup(src, "html.parser")
all_data = soup.find_all("div", class_="sc-1mo3ldo-0")
rest = all_data[6:]

for parent in rest:
    child = parent.find_all("div", class_="jumbo-tracker")
    for child2 in child:
        # get the link
        link_tag = child2.find("a")
        if 'href' in link_tag.attrs:
            link = link_tag.get('href')
            base = "https://www.zomato.com"
            link_new = urljoin(base, link)
            print(link_new)
        else:
            link_new = ''

        # get the rating
        rating_tag = child2.find("div", class_="sc-1q7bklc-1 cILgox")
        if rating_tag is not None:
            rating = rating_tag.text
            print(rating)
        else:
            rating = ''

        # get the name
        name_tag = child2.find("h4")
        if name_tag is not None:
            name = name_tag.text
            print(name)
        else:
            name = ''



        # get the cusine
        cusine_tag = child2.find_all("p", class_ ="sc-1hez2tp-0")
        cusine = ''
        for i in range(len(cusine_tag)):
            if 'for one' not in cusine_tag[i].text and i==3:
                Cusine = cusine_tag[i]
                cusine = Cusine.text
                print(cusine)
            elif "for one" in cusine_tag[i].text and i==3:
                Cusine = cusine_tag[i-1]
                cusine = Cusine.text
                print(cusine)


        # get the price 
        price_tags = child2.find_all("p", class_="sc-1hez2tp-0")
        for price_tag in price_tags:
            if "for one" in price_tag.text:
                one_person_price = price_tag.text
                print(one_person_price)
            

                
            
                        

        # write to csv
        row = [name, one_person_price, cusine, rating, link_new]
        with open('Table1.csv', 'a+', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        

    driver.quit()
    
    """

    # Display the code snippet
    st.code(code, language="python")
    
    st.title("Code for extracting details of Individual Restaraunt in Bangalore")

    # Define a code snippet
    code = """
    import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
import time 
import csv
df = pd.read_csv("C:/Users/karan/OneDrive/Desktop/Table1.csv")
df.head(5)
# open a new csv file with the name Table2
with open("Table2.csv", "a+", encoding="UTF8", newline='') as f:
    writer = csv.writer(f)
    header = ['Restaurant_Name', 'Latitude', 'Longitude', 'Location', 'Delivery_review_number']
    writer.writerow(header)

# Run a loop for getting each link open and get source file
for i in df.Urls:
    driver = webdriver.Chrome("chromedriver.exe")
    driver.get(i)
    time.sleep(1)
    src = driver.page_source
    soup = BeautifulSoup(src, "html.parser")


    #find the name of restaurent
    name = soup.find("h1", class_="sc-7kepeu-0 sc-iSDuPN fwzNdh")
    Name = name.text
    print(Name)


    # find delivery review number
    Del = soup.find_all("div", class_="sc-1q7bklc-8 kEgyiI")
    for i in range(len(Del)):
        if i == 1:
            Del = Del[i]
            Delivery = Del.text
            print(Delivery)


    # find location
    loc = soup.find("a", class_ = "sc-clNaTc vNCcy")
    Locat = loc.text
    print(Locat)


    #  find longtitude
    v = soup.find('a',rel="noopener noreferrer")
    link = v.get("href")
    Longi = link[-13:]
    print(Longi)


    # find latitude
    v1 = soup.find("a", rel="noopener noreferrer")
    link1 = v1.get("href")
    Lati = link1[-27:-14]
    print(Lati)



    row = [Name, Lati, Longi, Locat, Delivery]
    with open("Table2.csv", "a+", encoding="UTF8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    """

    # Display the code snippet
    st.code(code, language="python")

# Run the Streamlit app
if __name__ == "__main__":
    main()