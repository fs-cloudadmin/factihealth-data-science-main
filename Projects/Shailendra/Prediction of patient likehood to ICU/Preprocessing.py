
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#loading the data
df= pd.read_csv("admissions.csv.gz")
df_pat = pd.read_csv('patients.csv.gz')
df_diag = pd.read_csv('diagnoses_icd.csv.gz')



# function to find count previous_hospitalization

def previous_hospitalization(df):
    # no. of previous hospitalization
    df['admittime'] = pd.to_datetime(df['admittime'])
                                                             
    df = df.sort_values(by=['subject_id', 'admittime'])    # Sort DataFrame by subject_id and admittime
    
    last_admit_time = {}  # Dictionary to store the last admission time for each subject_id
    
    for index, row in df.iterrows():    # finding the last admission time for each subject_id
        subject_id = row['subject_id']
        last_admit_time[subject_id] = row['admittime']
        
    hospitalization_count = {}  #dictionary to store the hospitalization count
    
    for index, row in df.iterrows():   # finding the count of hospitalization
        subject_id = row['subject_id']
        if subject_id not in hospitalization_count:
            hospitalization_count[subject_id] = 0  
        if row['admittime'] > last_admit_time[subject_id]:
            hospitalization_count[subject_id] += 1
            
        last_admit_time[subject_id] = row['admittime']  # Update the last admission time for the current subject_id
        
        df.at[index, 'hospitalization_count'] = hospitalization_count[subject_id]  #hospitalization count for the current subject_id
    
    return df

# function to find length of stay

def length_of_stay(df):
    
    df['admittime'] = pd.to_datetime(df['admittime']) 
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['length_of_stay_days'] = (df['dischtime'] - df['admittime']).dt.days
    
    return df

def admission(df):
    
# handling admission type
    conditions = df['admission_type'].isin(['EW EMER.', 'URGENT', 'DIRECT EMER.'])
    df['admission_type'] = np.where(conditions, 'Emergency', df['admission_type'])
    
    conditions = df['admission_type'].isin(['OBSERVATION ADMIT', 'EU OBSERVATION', 'DIRECT OBSERVATION', 'AMBULATORY OBSERVATION'])
    df['admission_type'] = np.where(conditions, 'Observation', df['admission_type'])
    
    conditions = df['admission_type'].isin(['SURGICAL SAME DAY ADMISSION'])
    df['admission_type'] = np.where(conditions, 'Surgical Same Day Admission', df['admission_type'])
    
    conditions = df['admission_type'].isin(['ELECTIVE'])
    df['admission_type'] = np.where(conditions, 'Elective', df['admission_type'])
    
# admission location
    conditions = df['admission_location'].isin(['WALK-IN/SELF REFERRAL',"CLINIC REFERRAL","PROCEDURE SITE","PACU","INTERNAL TRANSFER TO OR FROM PSYCH",
                      "TRANSFER FROM SKILLED NURSING FACILITY","INFORMATION NOT AVAILABLE",
                      "AMBULATORY SURGERY TRANSFER"])
    df['admission_location'] = np.where(conditions, 'other', df['admission_location'])

# race
    df['race'] = np.where(df['race'].isin(['AMERICAN INDIAN/ALASKA NATIVE']), 'AMERICAN INDIAN/ALASKA NATIVE',
                            np.where(df['race'].isin(['ASIAN', 'ASIAN - ASIAN INDIAN', 'ASIAN - CHINESE',
                                                                           'ASIAN - KOREAN', 'ASIAN - SOUTH EAST ASIAN']),
                                     'ASIAN',
                            np.where(df['race'].isin(['BLACK/AFRICAN', 'BLACK/AFRICAN AMERICAN', 
                                                                           'BLACK/CAPE VERDEAN', 'BLACK/CARIBBEAN ISLAND']),
                                     'BLACK/AFRICAN',
                            np.where(df['race'].isin(['HISPANIC OR LATINO', 'HISPANIC/LATINO - CENTRAL AMERICAN',
                                                                           'HISPANIC/LATINO - COLUMBIAN', 'HISPANIC/LATINO - CUBAN',
                                                                           'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - GUATEMALAN', 
                                                                           'HISPANIC/LATINO - HONDURAN', 'HISPANIC/LATINO - MEXICAN', 
                                                                           'HISPANIC/LATINO - PUERTO RICAN','HISPANIC/LATINO - SALVADORAN']), 
                                     'HISPANIC OR LATINO',
                            np.where(df['race'].isin(['NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']), 
                                     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
                            np.where(df['race'].isin(['MULTIPLE RACE/race', 'OTHER', 
                                                                      'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN', 'UNKNOWN']), 
                                     'OTHER',
                            np.where(df['race'].isin(['PORTUGUESE']), 'PORTUGUESE',
                            np.where(df['race'].isin(['SOUTH AMERICAN']), 'SOUTH AMERICAN',
                            np.where(df['race'].isin(['WHITE', 'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN',
                                                                           'WHITE - OTHER EUROPEAN', 'WHITE - RUSSIAN']), 
                                     'WHITE',
                    np.nan)))))))))
    # Droping the useless columns to reduce frame size
    col_drop=[ 'admittime', 'dischtime', 'deathtime','admit_provider_id', 'discharge_location','language', 'marital_status',
       'edregtime', 'edouttime','hospital_expire_flag']
    df.drop(col_drop, axis=1, inplace= True)
    return df

def patient(df_pat):
    df_pat.drop(columns=['dod','anchor_year','anchor_year_group'], inplace=True)
    return df_pat


def diagnosis(df_diagnosis):
    df_diagnosis['icd'] = df_diagnosis.icd_code.str[:3]
    
    # categorize ICD codes
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
            
        # Apply function to create a new column
    df_diagnosis['diagnosis'] = df_diagnosis['icd'].apply(categorize_icd)
        
        # droping the not used attribute
    df_diagnosis.drop(columns=['icd_version','seq_num','icd','icd_code'], inplace=True)
    
        
        #Mapping the diagnosis of each hadm_id and patient_id
     
    pivoted_df = pd.pivot_table(df_diagnosis, index=['subject_id', 'hadm_id'], columns='diagnosis', aggfunc=lambda x: 1 if len(x) > 0 else 0, fill_value=0)
        # Reset index to make subject_id and hadm_id as columns
     
    pivoted_df.reset_index(inplace=True)
    return pivoted_df
  
    
def preprocessing(df, df_pat, df_diagcode):
    df1= previous_hospitalization(df)

    df2= length_of_stay(df1)

    df3= admission(df2)
    
    df_pat= patient(df_pat)
    
    df_diagcode= diagnosis(df_diagcode)
   
    
    adm_pat=pd.merge(df3,df_pat,on='subject_id',how='inner')## Merging the admisssion and patient detail
    
    adm_pat_diag=pd.merge(adm_pat,df_diagcode,on=['subject_id','hadm_id'],how='inner')  # merging the admission,patient and dianosis

    # droping the columns
    col_to_drop=['subject_id', 'hadm_id']
    
    adm_pat_diag.drop(columns=col_to_drop,inplace=True)
    # List of categorical columns
    categorical_columns = ['admission_type', 'insurance', 'race', 'gender','admission_location']

    # Perform one-hot encoding
    adm_pat_diag = pd.get_dummies(adm_pat_diag, columns=categorical_columns)
    
    #scaling 
    
    col_to_scale= [ 'hospitalization_count','length_of_stay_days','anchor_age']
    scaler = StandardScaler()
    adm_pat_diag[col_to_scale] = scaler.fit_transform(adm_pat_diag[col_to_scale])
    
    return adm_pat_diag
    
preprocessed_data = preprocessing(df, df_pat, df_diag)
preprocessed_data
