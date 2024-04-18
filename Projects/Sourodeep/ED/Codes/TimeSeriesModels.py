#!/usr/bin/env python
# coding: utf-8

# # Libraries

import pandas as pd
import pickle
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# edstays = pd.read_excel(r'Data/ed_data_scaled.xlsx')
edstays = pd.read_excel(r'Data/season_trend_data.xlsx')
edstays.head()


dataset = edstays.copy(deep=True)
# dataset['Date_Time_Admission'] = dataset['Date_Time_Admission'].apply(lambda x: x.replace(year=2021))
dataset['Date'] = pd.to_datetime(dataset['Date_Time_Admission']).dt.date
dataset['stay_duration'] = dataset['Discharge_Time'] - dataset['Date_Time_Admission']
dataset.head()

# Ensure that 'Date' column is in the datetime format and group patients by Date
data = dataset.groupby('Date')['Stay_ID'].count().reset_index()
data['Date'] = pd.to_datetime(data['Date']).dt.date
# Sort Date in ascending order
data.sort_values(by='Date', ascending=True, inplace=True)

## Get the Time Difference for Stay Duration
kpi_cards = dataset.groupby('Date')['stay_duration'].mean().reset_index()
kpi_cards['Admissions'] = dataset[dataset['Disposition']=='ADMITTED'
                                   ].groupby(['Date'])['Disposition'].count().reset_index()['Disposition']

original_df = pd.concat([data, kpi_cards[['stay_duration','Admissions']]], axis=1)
## Sort Date in ascending order
original_df.sort_values(by='Date', ascending=True, inplace=True)
# Set 'Date' column as the index# Set 'Date' column as the index
data.set_index('Date', inplace=True)
original_df.set_index('Date', inplace=True)

# Save the original data
# Pickle the Orginal dataset
# with open('Models\original_dataset.pkl', 'wb') as file:
#     pickle.dump(original_df, file)

# =============================================================================
# Start Model Training
# =============================================================================


# Define the number of days you want to forecast
forecast_days = 7

# Train-Test Split
train_data = data.iloc[:-forecast_days]
test_data = data.iloc[-forecast_days:]


def sarimax(model_dataset):
    # SARIMAX Model
    model_sarimax = SARIMAX(model_dataset, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_sarimax_fit = model_sarimax.fit(disp=False)
    
    # Make predictions for the next 7 days with SARIMAX
    sarimax_forecast = model_sarimax_fit.forecast(steps=forecast_days)
    return sarimax_forecast, model_sarimax_fit

sarimax_forecast, model_sarimax_fit = sarimax(train_data)

def prophet(model_dataset):
    # Prophet Model
    data_prophet = model_dataset.reset_index()
    data_prophet = data_prophet.rename(columns={'Date': 'ds', 'Stay_ID': 'y'})
    
    model_prophet = Prophet()
    model_prophet.fit(data_prophet)

    future = model_prophet.make_future_dataframe(periods=forecast_days)
    prophet_forecast = model_prophet.predict(future)
    
    prophet_forecast_values = prophet_forecast[['ds', 'yhat']].tail(forecast_days)
    
    return prophet_forecast_values, model_prophet

# Extract the forecasted values for the last 7 days
prophet_forecast_values, model_prophet = prophet(train_data)



# Evaluate SARIMAX Model
mae_sarimax = mean_absolute_error(test_data, sarimax_forecast)
mse_sarimax = mean_squared_error(test_data, sarimax_forecast)
rmse_sarimax = sqrt(mse_sarimax)

# Evaluate Prophet Model
# Ensure that the lengths of test_data and forecasted values match
test_data_prophet = test_data['Stay_ID'].values
forecast_prophet = prophet_forecast_values['yhat'].values

mae_prophet = mean_absolute_error(test_data_prophet, forecast_prophet)
mse_prophet = mean_squared_error(test_data_prophet, forecast_prophet)
rmse_prophet = sqrt(mse_prophet)

# Calculate MAPE for Prophet Model
def calculate_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    non_zero_indices = actual != 0
    actual_non_zero = actual[non_zero_indices]
    predicted_non_zero = predicted[non_zero_indices]
    mape = np.mean(np.abs((actual_non_zero - predicted_non_zero) / np.maximum(np.abs(actual_non_zero), 1))) * 100
    return mape


mape_sarimax = calculate_mape(test_data['Stay_ID'], sarimax_forecast)
mape_prophet = calculate_mape(test_data['Stay_ID'], prophet_forecast_values['yhat'])


# Assuming sarimax_forecast and test_data['Stay_ID'] contain the predicted and actual values, respectively
r2_sar = r2_score(test_data['Stay_ID'], sarimax_forecast)
r2_prp = r2_score(test_data['Stay_ID'], prophet_forecast_values['yhat'])


# Display the evaluation metrics
print("SARIMAX Model Evaluation:")
print(f"MAE: {mae_sarimax:.2f}")
print(f"MSE: {mse_sarimax:.2f}")
print(f"RMSE: {rmse_sarimax:.2f}")
print(f"MAPE: {mape_sarimax:.2f}%")
print("R-squared (R2) score:", r2_sar)

print("\nProphet Model Evaluation:")
print(f"MAE: {mae_prophet:.2f}")
print(f"MSE: {mse_prophet:.2f}")
print(f"RMSE: {rmse_prophet:.2f}")
print(f"MAPE: {mape_prophet:.2f}%")
print("R-squared (R2) score:", r2_prp)


# # Model Dump
# get the model on the entire Time period 

sarimax_forecast, model_sarimax_fit = sarimax(data)


# Assuming you have two models 'model1' and 'model2'
model1 = model_sarimax_fit
model2 = model_prophet
dataset = original_df

# Dump both models into a single file
with open('Models\combined_models.pkl', 'wb') as file:
    models = {'sarimax': model1, 'prophet': model2, 'dataset': dataset}
    pickle.dump(models, file)



# # # Pickle the Prophet model
# with open('prophet_model.pkl', 'wb') as file:
#     pickle.dump(model_prophet, file)


