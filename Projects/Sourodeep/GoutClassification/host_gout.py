# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:12:11 2024

@author: sourodeep.das
"""
import streamlit as st
import boto3
import json

# Specify the SageMaker endpoint name
endpoint_name = 'gout-prediction-v6'

# Function to make a prediction request to the SageMaker endpoint
def predict(input_text):
    # Specify the endpoint URL
    endpoint_url = f"https://runtime.sagemaker.ap-south-1.amazonaws.com/endpoints/{endpoint_name}/invocations"
    
    # Create a SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')
    
    # Invoke the endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/plain',
        Body=input_text.encode('utf-8')
    )

    # Check the HTTP status code
    status_code = response['ResponseMetadata']['HTTPStatusCode']

    if status_code == 200:
        try:
            # Attempt to parse the response as JSON
            result_json = response['Body'].read().decode('utf-8')
            result_dict = json.loads(result_json)
            return result_dict
        except json.JSONDecodeError as e:
            return f"Error decoding JSON: {e}"
    else:
        return f"Endpoint returned non-200 status code: {status_code}"

# Streamlit app
def main():
    st.title("Gout Prediction")

    # Input text area
    input_text = st.text_area("Enter your chief complaint here:",
                              """I have a family history of Gout, I have knee pain since last month.                                  
                              """)

    # Button to trigger prediction
    if st.button("Predict"):
        # Call the predict function
        result = predict(input_text)

        # Display the result
        if isinstance(result, dict):
            st.success("Prediction Result:")
            st.json(result)
        else:
            st.error(result)

if __name__ == "__main__":
    main()
