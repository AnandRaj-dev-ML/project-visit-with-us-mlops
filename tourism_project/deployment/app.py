import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Rajanan/model-visit-with-us-mlops", filename="best_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a Tourism Product Taken based on its operational parameters.
Please enter the specifiction data below to get a prediction.
""")

# User Input
age = st.number_input("Age", min_value=18, max_value=61, value=37)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=5, max_value=127, value=15)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons_visiting = st.selectbox("Number of Persons Visiting", [1, 2, 3,4,5])
num_followups = st.selectbox("Number of Followups", [1, 2, 3, 4, 5,6])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips", min_value=1, max_value=22, value=3)
passport = st.toggle("Has Passport")
pitch_satisfaction = st.selectbox("Pitch Satisfaction Score (1=Low, 5=High)", [1, 2, 3, 4, 5])
own_car = st.toggle("Owns a Car")
num_children_visiting = st.selectbox("Number of Children Visiting", [0, 1, 2,3])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=98678, value=22000, step=1000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_persons_visiting,
    'NumberOfFollowups': num_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': int(passport), # True/False into int
    'PitchSatisfactionScore': pitch_satisfaction,
    'OwnCar': int(own_car),   # True/False into int
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])


if st.button("Predict Product "):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
