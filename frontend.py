import streamlit as st
import requests

API_URL = "http://16.176.185.84:8000/predict" 

st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below:")


age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Are you a smoker?", options=[True, False])
city = st.text_input("City", value="Mumbai")
occupation = st.selectbox(
    "Occupation",
    ['retired', 'freelancer', 'student', 'government_job', 'business_owner', 'unemployed', 'private_job']
)

if st.button("Predict Premium Category"):
    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    try:
        response = requests.post(API_URL, json=input_data)
        result = response.json()

        if response.status_code == 200:

            prediction = result["predicted_category"]

            st.success("Prediction Completed")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Premium Category",
                    value=prediction["predicted_category"]
                )

            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{prediction['confidence']:.2%}"
                )

            st.subheader("Class Probabilities")

            probs = prediction["class_probabilities"]

            st.progress(float(probs["Low"]))
            st.write(f"🟢 Low: {probs['Low']:.2%}")

            st.progress(float(probs["Medium"]))
            st.write(f"🟡 Medium: {probs['Medium']:.2%}")

            st.progress(float(probs["High"]))
            st.write(f"🔴 High: {probs['High']:.2%}")

            st.subheader("Probability Distribution")

            st.bar_chart(probs)

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")