import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/header_images.jpg')
st.image(header_images)

# Add some information about the service
st.title("Air Contaminant Standard Index Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "air_data_form"):
    # Create select box input
    stasiun = st.selectbox(
        label = "1.\tFrom which station is this data collected?",
        options = (
            "DKI1 (Bunderan HI)",
            "DKI2 (Kelapa Gading)",
            "DKI3 (Jagakarsa)",
            "DKI4 (Lubang Buaya)",
            "DKI5 (Kebon Jeruk) Jakarta Barat"
        )
    )

    # Create box for number input
    pm10 = st.number_input(
        label = "2.\tEnter PM10 Value:",
        min_value = 0,
        max_value = 800,
        help = "Value range from 0 to 800"
    )
    
    pm25 = st.number_input(
        label = "3.\tEnter PM25 Value:",
        min_value = 0,
        max_value = 400,
        help = "Value range from 0 to 400"
    )

    so2 = st.number_input(
        label = "4.\tEnter SO2 Value:",
        min_value = 0,
        max_value = 500,
        help = "Value range from 0 to 500"
    )

    co = st.number_input(
        label = "5.\tEnter CO Value:",
        min_value = 0,
        max_value = 100,
        help = "Value range from 0 to 100"
    )

    o3 = st.number_input(
        label = "6.\tEnter O3 Value:",
        min_value = 0,
        max_value = 160,
        help = "Value range from 0 to 160"
    )

    no2 = st.number_input(
        label = "7.\tEnter NO2 Value:",
        min_value = 0,
        max_value = 100,
        help = "Value range from 0 to 100"
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "stasiun": stasiun,
            "pm10": pm10,
            "pm25": pm25,
            "so2": so2,
            "co": co,
            "o3": o3,
            "no2": no2
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://api:8080/predict", json = raw_data).json()

        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "BAIK":
                st.warning("Predicted Air Quality: TIDAK BAIK.")
            else:
                st.success("Predicted Air Quality: BAIK.")