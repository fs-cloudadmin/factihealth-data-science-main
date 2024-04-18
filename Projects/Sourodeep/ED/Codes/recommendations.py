# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:59:36 2023

@author: sourodeep.das
"""

import openai
# import pandas as pd

openai.api_key = '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'
def get_completion(prompt, model="gpt-4-1106-preview"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=0,
    )
    
    return response.choices[0].message["content"]


# prompt = """A hospital emergency department has 7 days of admission data with patient counts as  25-12-2021 with patient count as 405.0, 26-12-2021 with patient count as 448.0, 27-12-2021 with patient count as 182.0, 28-12-2021 with patient count as 585.0, 29-12-2021 with patient count as 493.0, 30-12-2021 with patient count as 542.0, 31-12-2021 with patient count as 315.0, and predicted patient counts for the next 7 days are 01-01-2022 with patient count as 467.0, 02-01-2022 with patient count as 459.0, 03-01-2022 with patient count as 445.0, 04-01-2022 with patient count as 387.0, 05-01-2022 with patient count as 428.0, 06-01-2022 with patient count as 431.0, 07-01-2022 with patient count as 471.0, . The hospital staff to patient ratio is 1:5 and total 100 beds are available in the department. Give insights and actions for each predicted days. Give a JSON output of the above text based on dates"""

# response = get_completion(prompt)
# print(response)


# date_list = ['01-01-2022', '02-01-2022', '03-01-2022', '04-01-2022', '05-01-2022', '06-01-2022', '07-01-2022']
# days_text = response.split("\n\n")  # Split by double newline assuming separation between days

# # Display text for each day separately
# for idx, day_text in enumerate(days_text):
#     if (idx==0 ) or (idx==len(days_text)-1):
#         continue
#     else:
#         print(f"Day {idx}:\n{day_text.strip()}\n")
        
        
def get_insights(prompt, pred_dates):
    
    response = get_completion(prompt, model="gpt-3.5-turbo")
    days_text = response.split("\n\n")
    
    return days_text
    
    
    
    
        
        
        