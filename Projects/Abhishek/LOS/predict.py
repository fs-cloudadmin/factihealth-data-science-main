import pandas as pd
import numpy as np

import pickle

from los_preprocess import los_preprocess
from redshift_read_table import read_table

def los_predict(df, model, verbose=True):

    hadm_ids = tuple(df['hadm_id'].unique())
    subject_ids = tuple(df['subject_ids'].unique())

    admission_query = "SELECT * FROM mimic.admissions WHERE hadm_id in {}".format(hadm_ids)
    patient_query = "SELECT * FROM mimic.patients WHERE subject_id in {}".format(subject_ids)
    diagnoses_icd_query = "SELECT * FROM mimic.diagnosis_icd WHERE hadm_id in {}".format(hadm_ids)
    icustays_query = "SELECT * FROM mimic.icustays WHERE hadm_id in {}".format(hadm_ids)

    df_adm = read_table(admission_query)
    df_pat = read_table(patient_query)
    df_diagcode = read_table(diagnoses_icd_query)
    df_icu = read_table(icustays_query)

    if verbose:
        print("(0/5) Data extracted from Redshift")

    df_hadm_id, df_clean, actual_median_los, actual_mean_los = los_preprocess(df=df_adm, 
                                                              df_pat=df_pat, 
                                                              df_diagcode=df_diagcode, 
                                                              df_icu=df_icu)

    # with open('/Users/abhishek/Documents/GitHub/Factspan/factihealth-data-science/Abhishek/LOS/los_model_xgboost.pkl', 'rb') as file:
    #     model = pickle.load(file)

    # if verbose:
    #     print("Model loaded")

    predictions = model.predict(df_clean)

    data = {
        'hadm_id': df_hadm_id,
        'predicted_los': predictions
    }

    los_predictions_df = pd.DataFrame(data)
    los_predictions_df['predicted_los'] = los_predictions_df['predicted_los'].round(0)

    if verbose:
        print("LOS Predictions done.")

    return los_predictions_df