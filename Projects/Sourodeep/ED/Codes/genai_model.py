# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:28:19 2023

@author: sourodeep.das
"""

from host_ed import call_combined_data

combined_data = call_combined_data()

data = {'Date': {0: '25-12-2021', 1: '26-12-2021', 2: '27-12-2021', 3: '28-12-2021',
                 4: '29-12-2021', 5: '30-12-2021', 6: '31-12-2021', 7: '01-01-2022',
                 8: '02-01-2022', 9: '03-01-2022', 10: '04-01-2022', 11: '05-01-2022',
                 12: '06-01-2022', 13: '07-01-2022'},
        'Predicted Count': {0: 405.0, 1: 448.0, 2: 182.0, 3: 585.0, 4: 493.0,
                            5: 542.0, 6: 315.0, 7: 467.0, 8: 459.0, 9: 445.0,
                            10: 387.0, 11: 428.0, 12: 431.0, 13: 471.0}}

# Extracting Date and Predicted Count columns as lists
dates = list(data['Date'].values())
predicted_counts = list(data['Predicted Count'].values())

# Combining Date and Predicted Count into a list of tuples
result = list(zip(dates, predicted_counts))


## Propmt creation
propmt = '''Give a summary of this information so that the hospital mangement can utilize and plan 
there resources related to staffing.A hospital emergency department has 7 days of admission data 
with patient counts as  '''
for i in result[:7]:
    propmt += str(i[0])+ ' with patient count as ' + str(i[1]) + ', '
    
propmt += 'and predicted patient counts for the next 7 days are '
for i in result[7:]:
    propmt += str(i[0])+ ' with patient count as ' + str(i[1]) + ', '
    




# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Load pre-trained GPT-2 model and tokenizer
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# # Your DataFrame or text input
# input_text = propmt
            
# ind = len(input_text)
# # Tokenize the input text
# input_ids = tokenizer.encode(input_text, return_tensors="pt")

# # Generate text using the model
# output = model.generate(input_ids, max_length=700, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

# # Decode and print the generated text
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)[ind:]
# print(generated_text)








