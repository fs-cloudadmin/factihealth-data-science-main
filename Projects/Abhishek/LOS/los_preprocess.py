import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")


def fill_missing_values(df, column_name, default_value=0):
    if column_name in df.columns:
        df[column_name].fillna(value=default_value, inplace=True)
    else:
        df[column_name] = 0


def los_preprocess(
    df,
    df_pat,
    df_diagcode,
    df_icu,
    verbose=True,
):
    """
    This function take 4 dataframes from the MIMIC-IV database,
    does cleanup and feature engineering for use in a Length-of-Stay
    regression model such as the sklearn GradientBoostingRegressor.

    INPUT:
    df - Primary Admissions information
    df_pat - Patient specific info such as gender and DOB
    df_diagcode - ICD Diagnosis for each admission to hospital
    df_icu - Intensive Care Unit (ICU) data for each admission

    OUTPUT:
    df - clean DataFrame for use in an regression model
    actual_median_los - Median los for all admissions
    actual_mean_los - Average los for all admissions
    """

    if verbose:
        print("(1/5) Completed dataframe imports")

    # Feature Engineering for Length of Stay (los) target variable
    # Convert admission and discharge times to datatime type
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    # Convert timedelta type into float 'days', 86400 seconds in a day
    df["los"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400

    # Drop columns that are not needed for next steps
    df.drop(
        columns=["dischtime", "edregtime", "edouttime", "hospital_expire_flag"],
        inplace=True,
    )

    # Track patients who died at the hospital by admission event
    df["deceased"] = df["deathtime"].notnull().map({True: 1, False: 0})

    # Hospital los metrics
    actual_mean_los = df["los"].loc[df["deceased"] == 0].mean()
    actual_median_los = df["los"].loc[df["deceased"] == 0].median()

    # Compress the number of race categories
    df["race"].replace(regex=r"^ASIAN\D*", value="ASIAN", inplace=True)
    df["race"].replace(regex=r"^WHITE\D*", value="WHITE", inplace=True)
    df["race"].replace(regex=r"^HISPANIC\D*", value="HISPANIC/LATINO", inplace=True)
    df["race"].replace(regex=r"^BLACK\D*", value="BLACK/AFRICAN AMERICAN", inplace=True)
    df["race"].replace(
        [
            "UNABLE TO OBTAIN",
            "OTHER",
            "PATIENT DECLINED TO ANSWER",
            "UNKNOWN/NOT SPECIFIED",
        ],
        value="OTHER/UNKNOWN",
        inplace=True,
    )
    df["race"].loc[
        ~df["race"].isin(df["race"].value_counts().nlargest(5).index.tolist())
    ] = "OTHER/UNKNOWN"

    # Clean the Admission Tyoe columns
    # Compress into EMERGENCY
    df["admission_type"].replace(to_replace="EW EMER.", value="EMERGENCY", inplace=True)
    df["admission_type"].replace(
        to_replace="DIRECT EMER.", value="EMERGENCY", inplace=True
    )
    df["admission_type"].replace(to_replace="URGENT", value="EMERGENCY", inplace=True)

    # Compress into EMERGENCY
    df["admission_type"].replace(
        to_replace="EU OBSERVATION", value="OBSERVATION", inplace=True
    )
    df["admission_type"].replace(
        to_replace="OBSERVATION ADMIT", value="OBSERVATION", inplace=True
    )
    df["admission_type"].replace(
        to_replace="DIRECT OBSERVATION", value="OBSERVATION", inplace=True
    )
    df["admission_type"].replace(
        to_replace="AMBULATORY OBSERVATION", value="OBSERVATION", inplace=True
    )

    # Re-categorize NaNs into 'Unknown'
    df["marital_status"] = df["marital_status"].replace("", np.nan)
    df["marital_status"] = df["marital_status"].fillna("UNKNOWN (DEFAULT)")

    if verbose:
        print("(2/5) Completed ADMISSIONS df cleanup and feature engineering.")

    # Feature Engineering for ICD9 code categories
    # Filter out E and V codes since processing will be done on the numeric first 3 values
    df_diagcode = df_diagcode[df_diagcode["icd_version"] == "9"]
    df_diagcode["recode"] = df_diagcode["icd_code"]
    df_diagcode["recode"] = df_diagcode["recode"][
        ~df_diagcode["recode"].str.contains("[a-zA-Z]").fillna(False)
    ]
    df_diagcode["recode"].fillna(value="999", inplace=True)
    df_diagcode["recode"] = df_diagcode["recode"].str.slice(start=0, stop=3, step=1)
    df_diagcode["recode"] = df_diagcode["recode"].astype(int)

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
    for num, cat_range in enumerate(icd9_ranges):
        df_diagcode["recode"] = np.where(
            df_diagcode["recode"].between(cat_range[0], cat_range[1]),
            num,
            df_diagcode["recode"],
        )

    # Convert integer to category name using diag_dict
    df_diagcode["recode"] = df_diagcode["recode"]
    df_diagcode["cat"] = df_diagcode["recode"].replace(diag_dict)

    # Create list of diagnoses for each admission
    hadm_list = df_diagcode.groupby("hadm_id")["cat"].apply(list).reset_index()

    # Convert diagnoses list into hospital admission-item matrix
    hadm_item = pd.get_dummies(hadm_list["cat"].apply(pd.Series).stack()).sum(level=0)

    # Join back with hadm_id, will merge with main admissions DF later
    hadm_item = hadm_item.join(hadm_list["hadm_id"], how="outer")

    # Merge with main admissions df
    df = df.merge(hadm_item, how="inner", on="hadm_id")

    if verbose:
        print("(3/5) Completed DIAGNOSES_ICD df cleanup and feature engineering.")

    # Feature Engineering for Age and gender
    df_pat = df_pat[["subject_id", "anchor_age", "gender"]]
    df = df.merge(df_pat, how="left", on="subject_id")

    # Create age categories
    age_ranges = [(0, 13), (13, 36), (36, 56), (56, 100)]
    for num, cat_range in enumerate(age_ranges):
        df["anchor_age"] = np.where(
            df["anchor_age"].between(cat_range[0], cat_range[1]), num, df["anchor_age"]
        )
    age_dict = {0: "newborn", 1: "young_adult", 2: "middle_adult", 3: "senior"}
    df["age"] = df["anchor_age"].replace(age_dict)

    # Re-map gender to boolean type
    df["gender"].replace({"M": 0, "F": 1}, inplace=True)

    if verbose:
        print("(4/5) Completed PATIENT df cleanup and feature engineering.")

    # Feature engineering for Intensive Care Unit (ICU) category
    # Reduce ICU categories to just ICU or NICU
    df_icu["first_careunit"].replace(
        {
            "Cardiac Vascular Intensive Care Unit (CVICU)": "Other-ICU",
            "Coronary Care Unit (CCU)": "Other-ICU",
            "Medical Intensive Care Unit (MICU)": "Other-ICU",
            "Medical/Surgical Intensive Care Unit (MICU/SICU)": "Other-ICU",
            "Neuro Stepdown": "Other-ICU",
            "Surgical Intensive Care Unit (SICU)": "Other-ICU",
            "Trauma SICU (TSICU)": "Other-ICU",
        },
        inplace=True,
    )
    df_icu["cat"] = df_icu["first_careunit"]
    icu_list = df_icu.groupby("hadm_id")["cat"].apply(list).reset_index()
    icu_item = pd.get_dummies(icu_list["cat"].apply(pd.Series).stack()).sum(level=0)
    icu_item[icu_item >= 1] = 1
    icu_item = icu_item.join(icu_list["hadm_id"], how="outer")
    df = df.merge(icu_item, how="left", on="hadm_id")

    if verbose:
        print("(5/5) Completed ICUSTAYS.csv cleanup and feature engineering.")

    # Remove deceased persons as they will skew los result
    df = df[df["deceased"] == 0]

    # Remove los with negative number, likely entry form error
    df = df[df["los"] > 0]

    df.drop_duplicates(inplace=True)

    # hadm_ids
    df_hadm_id = df["hadm_id"]

    # Drop unused columns, e.g. not used to predict los
    df.drop(
        columns=[
            "subject_id",
            "hadm_id",
            "admittime",
            "deathtime",
            "admit_provider_id",
            "admission_location",
            "discharge_location",
            "language",
            "deceased",
            "anchor_age",
        ],
        inplace=True,
    )

    prefix_cols = ["ADM", "INS", "RACE", "AGE", "MAR"]
    dummy_cols = ["admission_type", "insurance", "race", "age", "marital_status"]
    df = pd.get_dummies(df, prefix=prefix_cols, columns=dummy_cols)

    model_columns = [
        "los",
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
        "gender",
        "Neuro Intermediate",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)",
        "Other-ICU",
        "ADM_ELECTIVE",
        "ADM_EMERGENCY",
        "ADM_OBSERVATION",
        "ADM_SURGICAL SAME DAY ADMISSION",
        "INS_Medicaid",
        "INS_Medicare",
        "INS_Other",
        "RACE_ASIAN",
        "RACE_BLACK/AFRICAN AMERICAN",
        "RACE_HISPANIC/LATINO",
        "RACE_OTHER/UNKNOWN",
        "RACE_WHITE",
        "AGE_middle_adult",
        "AGE_senior",
        "AGE_young_adult",
        "MAR_DIVORCED",
        "MAR_MARRIED",
        "MAR_SINGLE",
        "MAR_UNKNOWN (DEFAULT)",
        "MAR_WIDOWED",
    ]

    for col in model_columns:
        fill_missing_values(df, col)

    df = df[['blood', 'circulatory', 'congenital', 'digestive', 'endocrine',
       'genitourinary', 'infectious', 'injury', 'mental', 'misc', 'muscular',
       'neoplasms', 'nervous', 'pregnancy', 'prenatal', 'respiratory', 'skin',
       'gender', 'Neuro Intermediate',
       'Neuro Surgical Intensive Care Unit (Neuro SICU)', 'Other-ICU',
       'ADM_ELECTIVE', 'ADM_EMERGENCY', 'ADM_OBSERVATION',
       'ADM_SURGICAL SAME DAY ADMISSION', 'INS_Medicaid', 'INS_Medicare',
       'INS_Other', 'RACE_ASIAN', 'RACE_BLACK/AFRICAN AMERICAN',
       'RACE_HISPANIC/LATINO', 'RACE_OTHER/UNKNOWN', 'RACE_WHITE',
       'AGE_middle_adult', 'AGE_senior', 'AGE_young_adult', 'MAR_DIVORCED',
       'MAR_MARRIED', 'MAR_SINGLE', 'MAR_UNKNOWN (DEFAULT)', 'MAR_WIDOWED']]

    if verbose:
        print("Data Preprocessing complete.")

    print(len(df_hadm_id))
    print(len(df))

    return df_hadm_id, df, actual_median_los, actual_mean_los