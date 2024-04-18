#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#loading the data
df_tri = pd.read_csv("triage.csv.gz")
df_diagnosis = pd.read_csv('diagnosis.csv.gz')



def fill_missing_values(df, column_name, default_value=0):
    if column_name in df.columns:
        df[column_name].fillna(value=default_value, inplace=True)
    else:
        df[column_name] = 0

        
        
def preprocess(
    df,
    df_diagnosis,
    df_tri,
)


    # Define columns to be normalized
    columns_to_normalize = ['temperature', 'heartrate', 'resprate', 'sbp', 'o2sat', 'dbp']

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit scaler to the data and transform the columns
    df_transform = df_tri.copy()
    df_transform[columns_to_normalize] = scaler.fit_transform(df_tri[columns_to_normalize])

    # Save the scaling parameters to a file (for future use)
    scaling_parameters = {
    'min': scaler.data_min_.tolist(),
    'max': scaler.data_max_.tolist(),
    'scale': scaler.scale_.tolist(),
    'columns': columns_to_normalize
    }
    
    return df_transform


    # Feature Engineering for ICD9 code categories
    # Filter out E and V codes since processing will be done on the numeric first 3 values
    df_diagnosis['recode'] = df_diagnosis['icd_code']
    df_diagnosis['recode'] = df_diagnosis['recode'][~df_diagnosis['recode'].str.contains("[a-zA-Z]").fillna(False)]
    df_diagnosis['recode'].fillna(value='999', inplace=True)

    # Take in consideration just the first 3 integers of the ICD9 code
    df_diagnosis['recode'] = df_diagnosis['recode'].str.slice(start=0, stop=3, step=1)
    df_diagnosis['recode'] = df_diagnosis['recode'].astype(int)

    # ICD-9 Main Category ranges
    icd9_ranges = [
        (1, 140),
        (140, 240),
        (240, 280),
        (280, 290),
        (290, 320),
        (320, 390),
        (390, 460),
        (460, 520),
        (520, 580),
        (580, 630),
        (630, 680),
        (680, 710),
        (710, 740),
        (740, 760),
        (760, 780),
        (780, 800),
        (800, 1000),
        (1000, 2000),
    ]

    # Associated category names
    diag_dict = {
        0: "infectious",
        1: "neoplasms",
        2: "endocrine",
        3: "blood",
        4: "mental",
        5: "nervous",
        6: "circulatory",
        7: "respiratory",
        8: "digestive",
        9: "genitourinary",
        10: "pregnancy",
        11: "skin",
        12: "muscular",
        13: "congenital",
        14: "prenatal",
        15: "misc",
        16: "injury",
        17: "misc",
    }

    # Re-code in terms of integer
    for num, category_range in enumerate(icd_ranges):
    df_diagnosis['recode'] = np.where(df_diagnosis['recode'].between(category_range[0],category_range[1]), num, df_diagnosis['recode'])
    
    # Convert integer to category name using diag_dict
    df_diagnosis['super_category'] = df_diagnosis['recode'].replace(diag_dict)
    
    # Create list of diagnoses for each stay
    stay_list = diagnosis.groupby('stay_id')['super_category'].apply(list).reset_index()
    
    diagnosis_item = pd.get_dummies(stay_list['super_category'].explode()).groupby(stay_list['stay_id']).sum()
    diagnosis_item.reset_index(inplace=True)
    
    # Merge with main triage df
    df = df.merge(df_transform, diagnosis_item, how="inner", on=['stay_id'], how='left')
    
    # Drop unused columns, e.g. not used to predict 
    df.drop(
        columns=[
            "pain",
            "acuity",
            "chiefcomplaint",
            "seq_num",
            "icd_version",
            "icd_title",
        ],
        inplace=True,
    )
    
    model_columns = [
        "blood",
        "circulatory",
        "congenital",
        "digestive",
        "endocrine",
        "genitourinary",
        "infectious",
        "injury",
        "mental",
        "misc",
        "muscular",
        "neoplasms",
        "nervous",
        "pregnancy",
        "prenatal",
        "respiratory",
        "skin",
    ]
    for col in model_columns:
        fill_missing_values(df, col)

    df = df[['blood', 'circulatory', 'congenital', 'digestive', 'endocrine',
       'genitourinary', 'infectious', 'injury', 'mental', 'misc', 'muscular',
       'neoplasms', 'nervous', 'pregnancy', 'prenatal', 'respiratory', 'skin',]
            
    print(len(df))
            
    return df
    
    


    
    
    
    
    

