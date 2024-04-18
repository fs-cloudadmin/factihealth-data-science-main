import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import StandardScaler


df=pd.read_csv("triage.csv.gz")

# Treating pain
non_numeric_values = pd.to_numeric(df['pain'], errors='coerce').isna()
df.loc[non_numeric_values, 'pain'] = None  # Replace non-numeric values with NaN

# Convert the 'pain' column to numeric
df['pain'] = pd.to_numeric(df['pain'], errors='coerce')

df.dropna(inplace=True)

# removing the outlier
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create a DataFrame of outliers
outliers_df = ((df < lower_bound) | (df > upper_bound))

# Filter the DataFrame to exclude outliers
df = df[~outliers_df.any(axis=1)]

# Reset the index of the filtered DataFrame
df.reset_index(drop=True, inplace=True)

column_to_drop = ['chiefcomplaint','subject_id','stay_id','acuity']
df = df.drop(columns=column_to_drop)

# normalizing the columns
columns_to_scale = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp','pain']
data_to_scale = df[columns_to_scale]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_scale)
df[columns_to_scale] = scaled_data

df
#import pickel file and test

# Postprocessing 
df=df-1   # This step is done to remap the output because we have map the target input