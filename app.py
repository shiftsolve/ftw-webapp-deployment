import altair as alt
import pandas as pd
import streamlit as st
import joblib
import numpy as np

def main():
    data = pd.read_csv('data/advertising_regression.csv')
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Raw Data","Exploration","Celebrate"])

    if page == "Homepage":
        #Title of your Web Application
        st.title('Sales Forecasting')

        #Describe you WebApp
        st.write('We demonstrate how we can forecast advertising sales based on ad expenditure.')
        pics = "download.png"
        st.image(pics, use_column_width=True)
        st.write("Please select a page on the left.")
       # st.write(data)
    
            
    elif page == "Raw Data":
        st.header("This is your Raw data.")
        st.write(data)
        #Radio
        st.subheader('Radio Advertising Cost Distribution')
        hist_values = np.histogram(data.radio, bins  = 300, range = (0,300))[0]
        st.bar_chart(hist_values)

        #TV
        st.subheader('TV Advertising Cost Distribution')
        hist_values = np.histogram(data.TV, bins  = 300, range = (0,300))[0]
        st.bar_chart(hist_values)

        #Newspaper
        st.subheader('Newspaper Advertising Cost Distribution')
        hist_values = np.histogram(data.newspaper, bins  = 300, range = (0,300))[0]
        st.bar_chart(hist_values)
        
        st.write("Please select a page on the left.")
    
        
    elif page == "Exploration":
        st.title("Data Exploration")
        TV = st.slider('TV Advertising Cost', 0 , 300, 150)
        radio = st.slider('Radio Advertising Cost', 0 , 50, 25)
        newspaper = st.slider('Newspaper Advertising Cost', 0 , 250, 125)
        

        
        #LOAD Saved Machine Learning model
        saved_model = joblib.load('advertising_model.sav')

        #Preedict sales using Variables/ features
        predicted_sales = saved_model.predict ([[TV, radio, newspaper]]) [0]
        st.write(f"## Predicted sales is {predicted_sales} dollars.")
        # 0 = min , 300 = max , 150 =  default
        
        st.write("\t")
        st.write("\t")
        st.write("\t")
        st.write("\t")
        st.write("\t")
        st.write("\t")
        st.write("\t")
        st.write("\t")
        st.write("Please select a page on the left.")
        
        
    elif page == "Celebrate":
        st.markdown("## Prediction Done")
        st.write("Yay! You're done knowing the predicted sales. Click below to celebrate.")
        btn = st.button("Celebrate!")
        if btn:
            st.balloons()
    

@st.cache
def load_data():
    data = data.cars()
    return data

def visualize_data(data, x_axis, y_axis):
    graph = alt.Chart(data).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis,
        color='Origin',
        tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    ).interactive()

    st.write(graph)

if __name__ == "__main__":
    main()