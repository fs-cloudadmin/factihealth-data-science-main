# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:51:30 2023

@author: sourodeep.das
"""

from recommendations import get_insights
import streamlit as st
import plotly.express as px
from datetime import timedelta, datetime
import pandas as pd
import pickle
import time
import json
pd.options.mode.chained_assignment = None

## Title
st.set_page_config(layout="wide")
st.title('Emergency Department Patients Admission')

# =============================================================================
# INPUT DATA
# =============================================================================

total_beds = 100
base_wait_time = timedelta(0,301,0)
staffs_to_patient = 10

def get_dates():
    ## Get dates for Start and End Date
    begin_date = datetime(2022, 1, 1)
    
    left_column, right_column, l1, l2 = st.columns(4)
    
    # Add date input widgets to each column
    with left_column:
        start_date = st.date_input("Enter the start date for the prediction",
                               value=begin_date, 
                               min_value=datetime(2022, 1, 1),
                               max_value=datetime(2022, 1, 1),
                               format="DD/MM/YYYY")
    
    with right_column:
        end_date = st.date_input("Enter the End date for the prediction",
                               value=start_date + timedelta(days=6), 
                               min_value=start_date,
                               max_value=begin_date+timedelta(days=6),
                               format="DD/MM/YYYY")
        
    st.markdown(f'''Prediction from Start Date: <b>{start_date.strftime("%d %b %Y")}</b> to
                End Date: <b>{end_date.strftime("%d %b %Y")}</b>''', unsafe_allow_html=True)
                
    return start_date, end_date
    
def date_diff(start_date, end_date):
    ## Get Date difference
    return (end_date-start_date).days + 1

# =============================================================================
# Load The Model
# =============================================================================

def get_predictions(start_date, end_date, algo):
    # Load the trained Prophet model
    # loaded_model = joblib.load('Models\prophet_model.pkl')  # Replace with the actual path to your saved model
    
    # =============================================================================
    # Create a new DataFrame with the date range you want to forecast
    new_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    if algo == 'prophet':
        
        new_df = pd.DataFrame({'ds': new_date_range})
        
        # Make forecasts for the new date range
        forecasts = model.predict(new_df)
        
        # Extract and print the forecasted values
        forecasted_values = forecasts[['ds', 'yhat']]
        forecasted_values = forecasted_values.rename(columns={'ds': 'Date', 'yhat': 'Patients Count'})
        forecasted_values['Patients Count'] = forecasted_values['Patients Count'].round(0).astype(int)
        forecasted_values['Date'] = forecasted_values['Date'].dt.date.astype(str)
        
    elif algo == 'sarimax':
        
        forecast = model.get_forecast(steps=hist_date)
        
        # Access the predicted values for the next 7 days
        predicted_values = forecast.predicted_mean
        # confidence_intervals = forecast.conf_int()
        forecasted_values = pd.DataFrame({'Date': new_date_range, 'Patients Count': predicted_values})
        forecasted_values['Patients Count'] = forecasted_values['Patients Count'].astype(int)

    return forecasted_values

def get_time(seconds):
    hours, minutes = divmod(seconds, 3600)
    minutes, seconds = divmod(minutes, 60)
    return hours, minutes

def get_stay_duration(combined_data):
    '''
    Calculate the Avg Stay duration
    '''
    ## Get the Average Stay Duration
    avg_time = combined_data['stay_duration'].mean()
    seconds = avg_time.seconds
    hours, minutes = get_time(seconds)
    # Format as "D:H:M"
    return hours, minutes

def get_admission_rate(combined_data):
    
    # combined_data['Admission Rate'] = combined_data['Admissions'] / combined_data['Predicted Count']
    add_rate = round(combined_data['Admission Rate'].mean()*100)
    return add_rate



def display_card(combined_data):
    '''
    This Block of Code Returns the KPI cards to the Dashboard

    '''
    
    kpi_lists = []
    
    date_row = st.columns(hist_date+1,)
    # date = combined_data.iloc[hist_date:]['Date'].tolist()
    date_list = [''] + combined_data.iloc[hist_date:]['Date'].tolist()
    ## Display Dates Row
    for i,v in enumerate(date_row):
        with date_row[i]:
            if i == 0:
                continue
            st.markdown(f'''<h5 style='text-align: center; background-color: #FFA500;
                      padding: 10px; border-radius: 10px; border: 1px solid #000'>{date_list[i]}</h5>''', unsafe_allow_html=True)

    st.write('')
    
    ## Patient Count Details
    patient_row = st.columns(hist_date+1)
    patients_count = combined_data.iloc[hist_date:]['Predicted Count'].astype(int).tolist()
    patients_count_list = ['Patients Count'] + patients_count
    
    for i,v in enumerate(patient_row):
        with patient_row[i]:
            # style = kpi_style('Hover Data', patients_count_list[i])
            st.write(f'''<h5 style='text-align: center; background-color: #fff0c1;
                      padding: 10px; border-radius: 10px; border: 1px solid #000; '>{patients_count_list[i]}</h5>''', unsafe_allow_html=True)
                
    
    st.write('')
    kpi_lists.append(patients_count_list)
    
    ## Bed Utilization
    bed_uti_row = st.columns(hist_date+1)
    hours, minutes = get_stay_duration(combined_data)
    converted_hours = hours + (minutes / 60)
    bed_utilization = [int((i * converted_hours) / (total_beds*24)*100) for i in patients_count]
    bed_utilization_list = ['Bed Utilization'] + bed_utilization
    
    for i,v in enumerate(bed_uti_row):
        with bed_uti_row[i]:
            st.write(f'''<h5 style='text-align: center; background-color: #fff0c1;
                      padding: 10px; border-radius: 10px; border: 1px solid #000; '>{bed_utilization_list[i]}%</h5>''', unsafe_allow_html=True)
                
    st.write('')
    kpi_lists.append(bed_utilization_list)
    
    ## Staffs Available
    staff_row = st.columns(hist_date+1)
    occupancy_rate = int(combined_data['Occupancy Rate'].mean()*100)
    staffs_required = [int(i/staffs_to_patient/occupancy_rate) for i in patients_count]
    staffs_required_list = ['Staffs Available'] + staffs_required
    
    for i,v in enumerate(staff_row):
        with staff_row[i]:
            st.write(f'''<h5 style='text-align: center; background-color: #fff0c1;
                      padding: 10px; border-radius: 10px; border: 1px solid #000; '>{staffs_required_list[i]}</h5>''', unsafe_allow_html=True)
    
    
    kpi_lists.append(staffs_required_list)
    
    st.write('\n\n')
    st.write('\n\n')
# =============================================================================
#     Header for KPI cards
# =============================================================================
    col = st.columns(4)

    with col[0]:
        st.markdown("<h5 style='text-align: center; color: black; background-color: #FFA500; border: 1px solid #000; padding: 10px; border-radius: 10px;'>Admissions to Ward</h5>", unsafe_allow_html=True)

    with col[1]:
        st.markdown("<h5 style='text-align: center; color: black; background-color: #FFA500; border: 1px solid #000; padding: 10px; border-radius: 10px;'>No. of Admissions</h5>", unsafe_allow_html=True)

    with col[2]:
        st.markdown("<h5 style='text-align: center; color: black; background-color: #FFA500; border: 1px solid #000; padding: 10px; border-radius: 10px;'>Wait Time</h5>", unsafe_allow_html=True)

    with col[3]:
        st.markdown("<h5 style='text-align: center; color: black; background-color: #FFA500; border: 1px solid #000; padding: 10px; border-radius: 10px;'>Stay Duration</h5>", unsafe_allow_html=True)

    st.write('\n\n')
    
# =============================================================================
#     Header for KPI cards Values
# =============================================================================
    col = st.columns(4)
    
    with col[0]:
        add_rate = get_admission_rate(combined_data)
        st.markdown(f'''<h2 style='text-align: center; color: black; background-color: #fff0c1; 
                    padding: 40px; border-radius: 10px; border: 1px solid #000; padding: 10px;'>{add_rate}% </h2>''', unsafe_allow_html=True)

    kpi_lists.append(add_rate)
    
    with col[1]:
        total_admin = combined_data.iloc[hist_date:]['Predicted Count'].sum() * add_rate/100
        st.markdown(f'''<h2 style='text-align: center; color: black; background-color: #fff0c1;
                    padding: 40px; border-radius: 10px; border: 1px solid #000; padding: 10px;'>{int(total_admin)}</h2>''', unsafe_allow_html=True)
                  
    kpi_lists.append(total_admin)                    

    with col[2]:
        wait_time = base_wait_time +  occupancy_rate*base_wait_time
        minutes, seconds = round(wait_time.seconds/60) , wait_time.seconds%60
        formatted_time = f"{minutes}m:{seconds:02d}s"
        st.markdown(f'''<h2 style='text-align: center; color: black; background-color: #fff0c1;
                    padding: 40px; border-radius: 10px; border: 1px solid #000; padding: 10px;'>{formatted_time}</h2>''', unsafe_allow_html=True)

    kpi_lists.append(formatted_time)       
    
    with col[3]:
        hours, minutes = get_stay_duration(combined_data)
        formatted_time = f"{hours}h:{minutes:02d}m"
        st.markdown(f'''<h2 style='text-align: center; color: black; background-color: #fff0c1;
                    padding: 40px; border-radius: 10px; border: 1px solid #000; padding: 10px;'>{formatted_time}</h2>''', unsafe_allow_html=True)

    kpi_lists.append(formatted_time)       

    st.write('*Note: Admissions are from Emergency Department to the Ward')
    st.write('\n\n')
    
    
    return kpi_lists
    
    
def get_predict_plotly(combined_data):
    
    # Create a combined bar and line chart using Plotly Express
    
    # hover_texts = [f"{data['date']}: {data['insight']}" for data in contents]

    ## Bar Chart
    fig = px.bar(combined_data, x="Date", y=["Actual Count"],
                 title="Actual vs Predicted Count Over Time",
                 labels='Historic Count',hover_data=None)  
    
    ## Line Chart
    fig.add_scatter(x=combined_data["Date"], y=combined_data["Predicted Count"], 
                    mode="lines+markers", line=dict(color="red"),
                    name="Predicted Values")
    px.bar()
    
    # Customize the layout
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Patient Count")
    
    # Set the background color of the entire plot
    fig.update_layout(
        plot_bgcolor="#fff0c1",  # Change the background color here
        paper_bgcolor="#fff0c1", # Change the paper (border) color here
        width=1000,  # Adjust the width of the figure
        height=600,  # Adjust the height of the figure
    )
    
    # Add text labels to the scatter points
    for i, row in combined_data.iterrows():
        fig.add_annotation(
            x=row["Date"],
            y=row["Predicted Count"]+15,
            text=str(int(row["Predicted Count"])),
            showarrow=False,
            font=dict(color="red", size=14),
            xshift=15
            )


    # # Display the chart using st.plotly_chart
    st.plotly_chart(fig)
    

def get_combined_data(historic_data):
    
    ## Get both History and Combined at same scale
    # historic_data['Date'] = historic_data['Date'].dt.strftime('%Y-%m-%d')
    # historic_data['Date'] = pd.to_datetime(historic_data['Date'])
    historic_data.rename(columns={'Stay_ID':'Patients Count'},inplace=True)
    
    combined_data = pd.DataFrame()
    combined_data = historic_data.copy(deep=True)
    combined_data['Actual Count'] = historic_data['Patients Count']
    combined_data.drop('Patients Count', inplace=True, axis=1)
    combined_data = pd.concat([combined_data, forecasted_values])
    combined_data.rename(columns={'Patients Count':'Predicted Count'},inplace=True)
    combined_data['Predicted Count'] = combined_data['Predicted Count'].fillna(combined_data['Actual Count'])
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])
    combined_data['Date'] = combined_data['Date'].dt.strftime('%d-%m-%Y')
    combined_data['Admission Rate'] = combined_data['Admissions']/combined_data['Predicted Count']
    combined_data['Occupancy Rate'] = combined_data['Admission Rate']*combined_data['Predicted Count']*(combined_data['stay_duration'].dt.seconds/60)/(total_beds*24)/100
    return combined_data

def create_prompt(combined_data, pred_dates, kpi_lists):
    
    dates = combined_data['Date'].tolist()
    predicted_counts = combined_data['Predicted Count'].tolist()

    # Combining Date and Predicted Count into a list of tuples
    result = list(zip(dates, predicted_counts))


    ## Propmt creation
    # Actual patient counts
    prompt = f'''A hospital emergency department has {len(pred_dates)} days of admission data with patient counts as  '''
    for i in result[:len(pred_dates)]:
        prompt += str(i[0])+ ' with patient count as ' + str(i[1]) + ', '
        
    # Predicted Counts
    prompt += f'and predicted patient counts for the next {len(pred_dates)} days are '
    for i in result[len(pred_dates):]:
        prompt += str(i[0])+ ' with patient count as ' + str(i[1]) + ', '
    
    # Bed Utilizations
    prompt += f'. Bed utilization in % for the {len(pred_dates)} days period is as follows'
    for i in kpi_lists[1][1:]:
        prompt += ', ' +str(i)
    prompt += '.'
    
    # Staffing Available
    prompt += f' Staffs required for the next {len(pred_dates)} days based on the Bed Occupancy is as follows:'
    for i in kpi_lists[2][1:]:
        prompt += ', ' +str(i)
    prompt += '.'
        
    prompt += ''' The hospital staff to patient ratio is 1:10 and total 100 beds are available in the department.
    Provide "action items", "recommendations" and "insights" as JSON output.
    '''
    return prompt

def display_bullet_points(category, content):
    st.write(f"### {category.capitalize()}")
    with st.expander(f"{category.capitalize()} Insights", expanded=False):
        for point in content:
            st.write(f"- {point}")

if __name__ == '__main__':
    ## Get the Input Start and End Date for Predictions
    start_date, end_date = get_dates()
    
    # Load the pickled Prophet model
    with open('Models\combined_models.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        
    original_dataset = loaded_model['dataset']
    
    ## Date Difference
    hist_date = date_diff(start_date, end_date)
    historic_data = original_dataset.iloc[-hist_date:].reset_index()

    ## Get the Forecasted Values
    model = loaded_model['sarimax']
    forecasted_values = get_predictions(start_date, end_date, 'sarimax')
    
    ## Get the Combined Data
    combined_data = get_combined_data(historic_data)
    pred_dates = combined_data['Date'].iloc[hist_date:].tolist()
    
    
    # Predict Button, On click displays values
    if st.button('Predict'):
        try:
            
            ## In Progress Button
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.001)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            
            ## Get the Display Cards (Either as a Card or as Markdown)
            kpi_lists = display_card(combined_data)
            
            # COnvert the layout into 4 parts, 1st 2 will be occupied by the chart. last will be by the Insights
            col1, col2, col3, col4 = st.columns([1, 1, 2, 3])
            
            # Show graph in first two columns
            with col1:
                get_predict_plotly(combined_data)
                
            
            # ## Create Propmt
            prompt = create_prompt(combined_data, pred_dates, kpi_lists)
            
            # ## API Call to Chat GPT
            insights = get_insights(prompt, pred_dates)
            
            # # Convert JSON string to Python dictionary
            contents = json.loads(insights[0])
                
            # In the last column, show the Insights such that It is a Dropdown Expander.
            ## Color code the background and display the body as bullet points
            with col4:
                with st.container():
                    insights_expander = st.expander("Insights", expanded=False)
                    points = contents['insights']
                    
                    with insights_expander:
                        html_content = """<div style='background-color: #FFA500; 
                                            padding: 10px; border-radius: 10px; height: 500px; overflow-y: auto;'>
                                            <h3>Insights</h3>
                                            <ul>"""
                        
                        for point in points:
                            html_content += f"<li><h5>{point}</h5></li>"
                        
                        html_content += "</ul></div>"
                        
                        st.markdown(html_content, unsafe_allow_html=True)
            
            
        except ValueError:
            st.write('**Invalid Date Range Selected**')
            
            
     
