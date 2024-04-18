# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:24:27 2024

@author: sourodeep.das
"""

import pandas as pd
import psycopg2
import time 

from sklearn.pipeline import Pipeline


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role
import seaborn as sns
import matplotlib.pyplot as plt


def db_connection(query):
    # Database connection parameters
    db_name = 'factihealth'   # Database name
    db_user = 'fh_user'  # Username
    db_password = 'Facti@874'  # Password
    db_host = 'redshift-cluster-factihealth.cuzgotkwtow6.ap-south-1.redshift.amazonaws.com'  # Cluster endpoint
    db_port = 5439  # Port
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        print("Connected to the database successfully")
        
        # Create a cursor object
        cur = conn.cursor()
        
        # Execute a query
        cur.execute(query)
        
        # Fetch all rows
        rows = cur.fetchall()
        
        if not rows:
            print("No rows returned from the query.")
            return None

        # Get column names from the cursor description
        col_names = [desc[0] for desc in cur.description]
        
        # Create a DataFrame from the fetched rows and column names
        df = pd.DataFrame(rows, columns=col_names)
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        
        return df
    except Exception as e:
        print(f"Database connection failed due to {e}")
        return None



# Example usage
query = 'Select * from mimic.gout_corpus'
data = db_connection(query)
dataset = data.copy(deep=True)
# Check if dataset is not None before using tail
display(dataset.tail(5))
## Remove the Unknown Data
dataset = dataset[~(dataset.predict=='-')]
print(dataset.shape)
dataset['predict'].unique()


entity_df = dataset.copy(deep=True)


# Assuming entity_df is your DataFrame
# If needed, replace 'chief_complaint' with the column name containing the text data
X = entity_df['chief_complaint']
y = entity_df['predict']

# Feature engineering - TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X_vectorized = tfidf_vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.8, random_state=42)


t1 = time.time()
from imblearn.over_sampling import SMOTE

# Assuming y_train and y_test are your target variables
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Apply SMOTE to oversample the minority classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train_encoded)

# Define the SVM classifier
svm_classifier = SVC(random_state=42)

# Define hyperparameters to search over
param_grid = {
    'C': [1],
    'kernel': ['rbf'],
}

# # Define hyperparameters to search over
# param_grid = {
#     'C': [0.1, 1,  5,  10],
#     'kernel': ['linear', 'rbf'],
# }

# Use GridSearchCV to search for the best hyperparameters
grid_search_svm = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, scoring='f1_macro', cv=5)
grid_search_svm.fit(X_resampled, y_resampled)

# Get the best parameters
best_params_svm = grid_search_svm.best_params_

# Train the final SVM model with the best parameters on the resampled data
final_model_svm = SVC(random_state=42, **best_params_svm)
final_model_svm.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred_test_svm = final_model_svm.predict(X_test)

# Evaluate the final SVM model on the test set
classification_report_result_svm = classification_report(y_test_encoded, y_pred_test_svm)



# Display results for the test set
print("SVM Test Set Classification Report:\n", classification_report_result_svm)
# Visualize the confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred_test_svm)
print((time.time() -t1)/60)


# Suppose new_sentence is the new text you want to predict
new_sentence = "No gout but pain in knees"

# Transform the new sentence using the TfidfVectorizer from the pipeline
new_sentence_vectorized = tfidf_vectorizer.transform([new_sentence])

# Use the trained SVM model to predict the class of the new sentence
predicted_class = final_model_svm.predict(new_sentence_vectorized)

# Suppose label_encoder is the LabelEncoder instance used during training
decoded_predicted_class = label_encoder.inverse_transform(predicted_class)

# Display the decoded predicted class
print("Decoded Predicted Class:", decoded_predicted_class[0])


# Display the predicted class
print("Predicted Class:", predicted_class[0])



