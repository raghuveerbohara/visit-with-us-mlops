
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("tourism_project/deployment/model.pkl")

st.title("Visit with Us - Wellness Tourism Package Prediction")

st.write(
    "Enter the customer details below to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

# Customer details
Age = st.number_input("Age", min_value=18, max_value=100, value=35)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox("City Tier", [1, 2, 3])

DurationOfPitch = st.number_input(
    "Duration of Pitch", min_value=0.0, value=15.0
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0.0,
    value=3.0
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3.0, 4.0, 5.0]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0.0,
    value=2.0
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0.0,
    value=0.0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=25000.0
)


# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [Age],
        "TypeofContact": [TypeofContact],
        "CityTier": [CityTier],
        "DurationOfPitch": [DurationOfPitch],
        "Occupation": [Occupation],
        "Gender": [Gender],
        "NumberOfPersonVisiting": [NumberOfPersonVisiting],
        "NumberOfFollowups": [NumberOfFollowups],
        "ProductPitched": [ProductPitched],
        "PreferredPropertyStar": [PreferredPropertyStar],
        "MaritalStatus": [MaritalStatus],
        "NumberOfTrips": [NumberOfTrips],
        "Passport": [Passport],
        "PitchSatisfactionScore": [PitchSatisfactionScore],
        "OwnCar": [OwnCar],
        "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
        "Designation": [Designation],
        "MonthlyIncome": [MonthlyIncome]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(
            "Prediction: Customer is likely to purchase the Wellness Tourism Package."
        )
    else:
        st.warning(
            "Prediction: Customer is unlikely to purchase the Wellness Tourism Package."
        )
