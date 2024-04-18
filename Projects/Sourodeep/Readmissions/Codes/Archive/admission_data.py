# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:23:05 2024

@author: sourodeep.das
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import psycopg2
from datetime import datetime


st.set_page_config(layout="wide")
st.title("Admission Data on Dates")

# Date selection sidebar
start_date = datetime(2024, 2, 1)
col = st.columns(4)
# Add the date selection widget to the first column
with col[0]:
    try:
        start_date, end_date = st.date_input("Select a Date Range", (start_date, start_date))
        search_button = st.button("Search")
    except ValueError:
        search_button=1
        st.write("Provide Start and End Date")


def db_connection(query):
    # Database connection parameters
    db_name = 'factihealth'   # Database name
    db_user = 'fh_user'  # Username
    db_password = 'Facti@874'  # Password
    db_host = 'redshift-cluster-factihealth.cuzgotkwtow6.ap-south-1.redshift.amazonaws.com'  # Cluster endpoint
    db_port = 5439  # Port
    # Connect to the database
    try:
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
        rows = cur.fetchall()
        
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
        
# Get Table Data based on table name
def get_db_data(table_name, patients_ids=()):
    ''' Get Data based on the table name provided '''
    # query = f"""SELECT * FROM factihealth.mimic.{table_name} 
    #             WHERE subject_id in {patients_ids}
    #             """
    query = f"SELECT * FROM factihealth.mimic.{table_name}"
    if patients_ids:
        query += f" WHERE subject_id IN {patients_ids}"
    df = db_connection(query)
    return df


# Function to categorize ICD codes
def categorize_icd(icd):
    try:
        if 1 <= int(icd) <= 139:
            return 'Infectious and Parasitic Diseases'
        elif 140 <= int(icd) <= 239:
            return 'Neoplasms'
        elif 240 <= int(icd) <= 279:
            return 'Endocrine, Nutritional, and Metabolic Diseases'
        elif 280 <= int(icd) <= 289:
            return 'Diseases of the Blood and Blood-forming Organs'
        elif 290 <= int(icd) <= 319:
            return 'Mental Disorders'
        elif 320 <= int(icd) <= 389:
            return 'Diseases of the Nervous System and Sense Organs'
        elif 390 <= int(icd) <= 459:
            return 'Diseases of the Circulatory System'
        elif 460 <= int(icd) <= 519:
            return 'Diseases of the Respiratory System'
        elif 520 <= int(icd) <= 579:
            return 'Diseases of the Digestive System'
        elif 580 <= int(icd) <= 629:
            return 'Diseases of the Genitourinary System'
        elif 630 <= int(icd) <= 679:
            return 'Complications of Pregnancy, Childbirth, and the Puerperium'
        elif 680 <= int(icd) <= 709:
            return 'Diseases of the Skin and Subcutaneous Tissue'
        elif 710 <= int(icd) <= 739:
            return 'Diseases of the Musculoskeletal System and Connective Tissue'
        elif 740 <= int(icd) <= 759:
            return 'Congenital Anomalies'
        elif 760 <= int(icd) <= 779:
            return 'Certain Conditions Originating in the Perinatal Period'
        elif 780 <= int(icd) <= 799:
            return 'Symptoms, Signs, and Ill-defined Conditions'
        elif 800 <= int(icd) <= 999:
            return 'Injury and Poisoning'
        
    except ValueError:
        if icd[0] == 'A' or icd[0] == 'B':
            return 'Infectious and Parasitic Diseases'
        elif icd[0] == 'C':
            return 'Neoplasms'
        elif icd[0] == 'D':
            return 'Diseases of the Blood and Blood-forming Organs'
        elif icd[0] == 'E':
            return 'External causes of injury and supplemental classification'
        elif icd[0] == 'F':
            return 'Mental Disorders'
        elif icd[0] == 'G':
            return 'Diseases of the Nervous System and Sense Organs'
        elif icd[0] == 'H':
            return 'Diseases of the eye, adnexa and mastoid process'
        elif icd[0] == 'I':
            return 'Diseases of the Circulatory System'
        elif icd[0] == 'J':
            return 'Diseases of the Respiratory System'
        elif icd[0] == 'K':
            return 'Diseases of the Digestive System'
        elif icd[0] == 'L':
            return 'Diseases of the Skin and Subcutaneous Tissue'
        elif icd[0] == 'M':
            return 'Diseases of the Musculoskeletal System and Connective Tissue'
        elif icd[0] == 'N':
            return 'Diseases of the Genitourinary System'
        elif icd[0] == 'O':
            return 'Complications of Pregnancy, Childbirth, and the Puerperium'
        elif icd[0] == 'P':
            return 'Certain Conditions Originating in the Perinatal Period'
        elif icd[0] == 'Q':
            return 'Congenital Anomalies'
        elif icd[0] == 'R':
            return 'Symptoms, Signs, and Ill-defined Conditions'
        elif icd[0] == 'S' or icd[0] == 'T' :
            return 'Injury and Poisoning'
        elif icd[0] == 'U':
            return 'Codes for special purposes'
        elif icd[0] in ['V', 'W', 'X' ,'Y']:
            return 'External causes of injury and supplemental classification'
        elif icd[0] == 'Z':
            return 'Factors influencing health status and contact with health services'
                
        
# =============================================================================
# Load All the Data
# =============================================================================

def load_data():
    ## Admissions
    admissions_df = get_db_data('admissions_df', ())
    admissions_df['date'] = admissions_df['admittime'].dt.date
    
    admissions_df = admissions_df[(admissions_df['date'] >= start_date) &
                                  (admissions_df['date'] <= end_date)]
    ## Get Patients Information
    patients_ids = tuple(admissions_df.subject_id.unique())
    
    ## Diagnosis ICD
    diagnosis_icd_df = get_db_data('diagnosis_icd_df', patients_ids)
    diagnosis_icd_df = diagnosis_icd_df[diagnosis_icd_df['subject_id'].isin(patients_ids)]
    
    
    ## ICU Stays
    icustays_df = get_db_data('icustays_df')
    icustays_df = icustays_df[icustays_df['subject_id'].isin(patients_ids)]
    ## Chart Events
    chartevents_df = get_db_data('chartevents_df')
    chartevents_df = chartevents_df[chartevents_df['subject_id'].isin(patients_ids)]
    ## Patient Data
    patients_df = get_db_data('patients', patients_ids)
    
    return admissions_df, diagnosis_icd_df, icustays_df, chartevents_df, patients_df
    

def categorize_race(admin_diag):
    ## Recategorize Race
    admin_diag['race'] = np.where(admin_diag['race'].isin(['AMERICAN INDIAN/ALASKA NATIVE']), 'AMERICAN INDIAN/ALASKA NATIVE',
                            np.where(admin_diag['race'].isin(['ASIAN', 'ASIAN - ASIAN INDIAN', 'ASIAN - CHINESE',
                                                                           'ASIAN - KOREAN', 'ASIAN - SOUTH EAST ASIAN']),
                                     'ASIAN',
                            np.where(admin_diag['race'].isin(['BLACK/AFRICAN', 'BLACK/AFRICAN AMERICAN', 
                                                                           'BLACK/CAPE VERDEAN', 'BLACK/CARIBBEAN ISLAND']),
                                     'BLACK/AFRICAN',
                            np.where(admin_diag['race'].isin(['HISPANIC OR LATINO', 'HISPANIC/LATINO - CENTRAL AMERICAN',
                                                                           'HISPANIC/LATINO - COLUMBIAN', 'HISPANIC/LATINO - CUBAN',
                                                                           'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - GUATEMALAN', 
                                                                           'HISPANIC/LATINO - HONDURAN', 'HISPANIC/LATINO - MEXICAN', 
                                                                           'HISPANIC/LATINO - PUERTO RICAN','HISPANIC/LATINO - SALVADORAN']), 
                                     'HISPANIC OR LATINO',
                            np.where(admin_diag['race'].isin(['NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']), 
                                     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
                            np.where(admin_diag['race'].isin(['MULTIPLE RACE/race', 'OTHER', 
                                                                      'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN', 'UNKNOWN']), 
                                     'OTHER',
                            np.where(admin_diag['race'].isin(['PORTUGUESE']), 'PORTUGUESE',
                            np.where(admin_diag['race'].isin(['SOUTH AMERICAN']), 'SOUTH AMERICAN',
                            np.where(admin_diag['race'].isin(['WHITE', 'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN',
                                                                           'WHITE - OTHER EUROPEAN', 'WHITE - RUSSIAN']), 
                                     'WHITE',
                                     np.nan)))))))))
    

    return admin_diag


def pre_process_readmission():
    
# =============================================================================
#     GET Patient and Diagnosis Data
# =============================================================================

    admissions_df, diagnosis_icd_df, icustays_df, chartevents_df, patients_df = load_data()


    ## Diagonsis Data Clean
    diagnosis_icd_df['icd'] = diagnosis_icd_df.icd_code.str[:3]
    diagnosis_icd_df['diagnosis'] = diagnosis_icd_df['icd'].apply(categorize_icd)
    diagnosis_icd_df = diagnosis_icd_df[['subject_id', 'hadm_id', 'diagnosis']]
    
    ## Merge Diagnosis Data with Diagnosis Code(ICD)
    patients_df = patients_df[['subject_id', 'gender', 'anchor_age']]
    pat_diagn_data = pd.merge(diagnosis_icd_df, patients_df, on='subject_id', how='left')
    pat_diagn_data.drop_duplicates(inplace=True)
    
# =============================================================================
#     Merge Admission Data with Diagnosis Data
# =============================================================================
    admin_diag = pd.merge(admissions_df, pat_diagn_data, on=['subject_id','hadm_id'], 
                          how='left')

    ## Select the required Columns
    admin_diag = admin_diag[['subject_id', 'hadm_id', 'admittime','admission_type',
                             'admission_location','discharge_location', 'insurance', 
                             'language', 'marital_status', 'race',
                             'gender', 'anchor_age', 'diagnosis']]
    
    ## Convert Time to Date
    admin_diag['admittime'] = pd.to_datetime(admin_diag['admittime'])
    admin_diag['admitdate'] = admin_diag['admittime'].dt.date
    
    # Sort Values on the admin time
    admin_diag.sort_values(['subject_id', 'admittime'], inplace=True)
    
    ## Get the time difference
    admin_diag['admitdate_diff'] = admin_diag.groupby('subject_id')['admittime'].diff()

    # Replace 0s with NaN
    admin_diag['admitdate_diff'] = admin_diag['admitdate_diff'].replace(pd.Timedelta(0), np.nan)
    
    # Forward fill NaN values
    admin_diag['admitdate_diff'] = admin_diag['admitdate_diff'].fillna(method='ffill')
    
    # Define boolean conditions for each range
    admin_diag['<30'] = admin_diag['admitdate_diff'] <= pd.Timedelta(days=30)
    admin_diag['<60'] = (admin_diag['admitdate_diff'] <= pd.Timedelta(days=60))
    admin_diag['>365'] = admin_diag['admitdate_diff'] > pd.Timedelta(days=365)
    
    # Convert boolean values to integers (0 and 1)
    admin_diag['<30'] = admin_diag['<30'].astype(int)
    admin_diag['<60'] = admin_diag['<60'].astype(int)
    admin_diag['>365'] = admin_diag['>365'].astype(int)
    ## Convert to Days
    admin_diag['admitdate_diff'] = admin_diag['admitdate_diff'].dt.days
    
# =============================================================================
#    APPLIED RULES
# =============================================================================
    ## Recategorize Admission Type
    conditions = admin_diag['admission_type'].isin(['EW EMER.', 'URGENT', 'DIRECT EMER.'])
    admin_diag['admission_type'] = np.where(conditions, 'Emergency', admin_diag['admission_type'])
    
    conditions = admin_diag['admission_type'].isin(['OBSERVATION ADMIT', 'EU OBSERVATION', 'DIRECT OBSERVATION', 'AMBULATORY OBSERVATION'])
    admin_diag['admission_type'] = np.where(conditions, 'Observation', admin_diag['admission_type'])
    
    conditions = admin_diag['admission_type'].isin(['SURGICAL SAME DAY ADMISSION'])
    admin_diag['admission_type'] = np.where(conditions, 'Surgical Same Day Admission', admin_diag['admission_type'])
    
    conditions = admin_diag['admission_type'].isin(['ELECTIVE'])
    admin_diag['admission_type'] = np.where(conditions, 'Elective', admin_diag['admission_type'])
    
# =============================================================================
#     CATEGPROIZE RACES
# =============================================================================
    admin_diag = categorize_race(admin_diag)
    
# =============================================================================
#     Merge Admission Data with Chart events
# =============================================================================
    master_merged = pd.merge(admin_diag, chartevents_df, on=['hadm_id','subject_id'], how='left')
    master_merged = master_merged.fillna(0)
    
    ## Select the required columns
    columns= ['subject_id', 'hadm_id', 'admittime', 'admission_type', 'admission_location',
              'insurance', 'language', 'marital_status', 'race','diagnosis', 'gender',
              'anchor_age','admitdate', 'admitdate_diff','valuenum', 'valueuom', 
              '<30', '<60', '>365']
    master_merged = master_merged[columns]
    
    
    master_merged.subject_id.nunique()
    master_merged.hadm_id.nunique()
    
    ## Explode the columns which have the same values

    explode_colomns = ['anchor_age', 'admission_type','gender','admission_location', 
                       'insurance', 'language', 'marital_status', 'race', 'diagnosis', 'admitdate_diff', '<30', '<60', '>365']
    master_merged_df = master_merged.groupby(['subject_id', 'hadm_id','valuenum', 'valueuom'])[explode_colomns].agg(list).reset_index()

    ## Convert 'Nan' to NaN
    master_merged_df['race'] = master_merged_df['race'].replace('nan', np.nan)
    
   # Apply the lambda function to each specified columnto get the unique value in the list
    ## This is to ensure same column have the unique value. eg. [EMERGENCY ROOM, EMERGENCY ROOM,....] to EMERGENCY ROOM
    ## Diagnosis is been considered because it may or may not have unique values.
    
    for col in explode_colomns:
        if col!= 'diagnosis':
            master_merged_df[col] = master_merged_df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

    ## Convert 'Nan' to NaN
    master_merged_df['race'] = master_merged_df['race'].replace('nan', np.nan)


# =============================================================================
#     Create Dummies
# =============================================================================
    # Perform one-hot encoding for nominal categorical variables
    cat_columns=['admission_type', 'admission_location', 'insurance', 'language',
                 'marital_status', 'race', 'gender', 'valueuom']
    
    master_data = pd.get_dummies(master_merged_df, columns=cat_columns)
    
    dummy_df = pd.get_dummies(master_merged_df['diagnosis'].explode(), prefix='diagnosis')
    master_data = pd.concat([master_merged_df, dummy_df.groupby(level=0).max()], axis=1)
    master_data.drop(columns=['diagnosis'], inplace=True)
    print(master_data.shape)

    # ## Check for all the columns present
    # for i in ['valueuom', 'admission_type', 'gender', 'admission_location', 
    #          'insurance', 'language', 'marital_status', 'race'  ]:
    #     print(i, master_data[i].nunique())    
        
        
    cat_columns=['valueuom', 'admission_type', 'gender', 'admission_location', 
             'insurance', 'language', 'marital_status', 'race'  ]


    master_data = pd.get_dummies(master_data, columns=cat_columns, drop_first=True)
    print(master_data.shape)
    master_data.dropna(inplace=True)
    print(master_data.shape)

    
    
# =============================================================================
#     Create the exact same no of columns as present in the Model Building
# =============================================================================
    all_columns = ['subject_id', 'hadm_id', 'valuenum', 'anchor_age', 'admitdate_diff',
                   '<30', '<60', '>365',
                   'diagnosis_Certain Conditions Originating in the Perinatal Period',
                   'diagnosis_Complications of Pregnancy, Childbirth, and the Puerperium',
                   'diagnosis_Congenital Anomalies',
                   'diagnosis_Diseases of the Blood and Blood-forming Organs',
                   'diagnosis_Diseases of the Circulatory System',
                   'diagnosis_Diseases of the Digestive System',
                   'diagnosis_Diseases of the Genitourinary System',
                   'diagnosis_Diseases of the Musculoskeletal System and Connective Tissue',
                   'diagnosis_Diseases of the Nervous System and Sense Organs',
                   'diagnosis_Diseases of the Respiratory System',
                   'diagnosis_Diseases of the Skin and Subcutaneous Tissue',
                   'diagnosis_Diseases of the eye, adnexa and mastoid process',
                   'diagnosis_Endocrine, Nutritional, and Metabolic Diseases',
                   'diagnosis_External causes of injury and supplemental classification',
                   'diagnosis_Factors influencing health status and contact with health services',
                   'diagnosis_Infectious and Parasitic Diseases',
                   'diagnosis_Injury and Poisoning', 'diagnosis_Mental Disorders',
                   'diagnosis_Neoplasms',
                   'diagnosis_Symptoms, Signs, and Ill-defined Conditions', 'valueuom_bpm',
                   'valueuom_insp/min', 'valueuom_kg', 'valueuom_mg/dL', 'valueuom_mmHg',
                   'valueuom_units', 'valueuom_°F', 'admission_type_Emergency',
                   'admission_type_Observation',
                   'admission_type_Surgical Same Day Admission', 'gender_M',
                   'admission_location_CLINIC REFERRAL',
                   'admission_location_EMERGENCY ROOM',
                   'admission_location_INFORMATION NOT AVAILABLE',
                   'admission_location_INTERNAL TRANSFER TO OR FROM PSYCH',
                   'admission_location_PACU', 'admission_location_PHYSICIAN REFERRAL',
                   'admission_location_PROCEDURE SITE',
                   'admission_location_TRANSFER FROM HOSPITAL',
                   'admission_location_TRANSFER FROM SKILLED NURSING FACILITY',
                   'admission_location_WALK-IN/SELF REFERRAL', 'insurance_Medicare',
                   'insurance_Other', 'language_ENGLISH', 'marital_status_MARRIED',
                   'marital_status_SINGLE', 'marital_status_WIDOWED', 'race_ASIAN',
                   'race_BLACK/AFRICAN', 'race_HISPANIC OR LATINO',
                   'race_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'race_OTHER',
                   'race_PORTUGUESE', 'race_SOUTH AMERICAN', 'race_WHITE']
    
    master_columns = master_data.columns
    
    # Filter out columns from all_columns that are not present in master_columns
    missing_columns = [column for column in all_columns if column not in master_columns]
    # Fill missing columns with False
    for column in missing_columns:
        master_data[column] = False
    
    ## Rearrange the columns
    master_data = master_data[all_columns]
    master_data = master_data.drop(['subject_id', 'hadm_id', 'admitdate_diff',
                                    '<30', '<60', '>365'], axis=1)
    
    master_data.drop_duplicates(inplace=True)
    return master_data, master_merged_df
    


def load_model(master_data):
    # Load the pickled model
    
    forecasts = pd.DataFrame()
    with open('Models/readmission_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        
    for key, model_path in loaded_model.items():
        forecasts[key] = model_path.predict(master_data)
        
    return forecasts
        
try:
    if search_button and start_date:
        
        master_data, master_merged_df = pre_process_readmission()
        
        forecasts = load_model(master_data)
        
        # forecasts = pd.DataFrame({'<30 Days': forecasts_30,
        #                           '<60 Days': forecasts_60,
        #                           '>365 Days': forecasts_365
        #                           })
        
        final = pd.concat([master_merged_df[['subject_id', 'hadm_id', 'anchor_age',
                                             'admission_type', 'gender', 'admission_location',
                                             'marital_status', 'race']], 
                           forecasts], axis=1)
        # Display filtered data
        st.subheader(f"Admissions from {start_date} to {end_date}")
        
        df = final.groupby(['subject_id', 'hadm_id', 'anchor_age',
                        'admission_type', 'gender', 'admission_location',
                        'marital_status', 'race'])[['30_days','60_days','365_days']].max()

        st.write(df)
    elif search_button and not start_date:
        st.warning("Please select a date before searching.")
except: 
    st.write("Unexpected Error")

