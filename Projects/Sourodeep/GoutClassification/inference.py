# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:03:12 2024

@author: sourodeep.das
"""

import os
import json
import joblib
import logging

logging.basicConfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')


def model_fn(model_dir=os.getcwd()):
    # Load the model directly from the joblib file
    model_path = os.path.join(model_dir, "gout_pipeline.joblib")
    
    logging.info('MODEL FILE PATH {}'.format(model_path))
    
    loaded_model = joblib.load(model_path)
    logging.info('LOADING SUCCESSFUL')

    return loaded_model

# Define the input_fn function for input data processing
def input_fn(request_body, request_content_type):
    # Assuming the input is a plain text string
    logging.info('Entering Input Function')
    return [request_body]

# Define the predict_fn function for making predictions
def predict_fn(input_data, model):
    logging.info('Entering Predict Function')
    # Input data is a list with a single text element
    logging.info('INPUT DATA {}'.format(input_data))
    predictions = model.transform(input_data)
    logging.info('PREDICTION POST TRANSFORM {}'.format(predictions))
    logging.info('PREDICTION Type {}'.format(type(predictions[0][0])))
    return predictions[0][0]

# Define the output_fn function for output data processing
def output_fn(prediction, accept):
    # Assuming your output is a simple array of predictions, modify this based on your output format
    logging.info('OUTPUT PREDICTION {}'.format(prediction))
    
    # Convert prediction to JSON format
    json_output = json.dumps(prediction)
    logging.info('JSON OUTPUT {}'.format(json_output))
    return json_output

# Example main function (not used in SageMaker endpoint, only for local testing)
if __name__ == "__main__":
    # Load the model
    model = model_fn(os.getcwd())

    # Example input data for testing
    input_data = "Your text input goes here."

    # Preprocess input data
    input_data = input_fn(input_data, 'text/plain')

    # Make predictions
    predictions = predict_fn(input_data, model)

    # Post-process predictions if necessary
    output_data = output_fn(predictions, "application/json")

    print(f"Predictions: {output_data}")
