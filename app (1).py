
import streamlit as st
import joblib
import pandas as pd

# 1. Set the title of the Streamlit application
st.title('Loan Repayment Prediction')

# 2. Load the previously saved logistic_regression_model.joblib
model = joblib.load('logistic_regression_model.joblib')
st.write("Model loaded successfully!")

# 3. Create a sample input DataFrame for prediction
# In a real app, users would input values via widgets

# For this demonstration, we are creating a dummy input for now.
# In a real Streamlit app, you would gather input from the user for each feature.

# Get feature names from the trained model
feature_names = model.feature_names_in_

sample_input_data = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

st.write("### Please input feature values to make a prediction:")

# Display the dummy input for user reference (in a deployed app, these would be interactive widgets)
st.markdown("*(Note: In a fully interactive app, input fields for all features would be present here.)*
")
st.dataframe(sample_input_data.head())

# 4. Use the loaded model to make a prediction
if st.button('Predict Loan Repayment'):
    prediction = model.predict(sample_input_data)
    prediction_proba = model.predict_proba(sample_input_data)

    # 5. Display the prediction result
    st.write("
--- Prediction Result ---")
    if prediction[0] == 0:
        st.success(f"The loan is likely to be repaid. (Probability of 0: {prediction_proba[0][0]:.2f})")
    elif prediction[0] == 1:
        st.error(f"The loan is likely NOT to be repaid. (Probability of 1: {prediction_proba[0][1]:.2f})")
    
st.write("This is a basic demonstration. In a full application, users would interactively provide input values.")
