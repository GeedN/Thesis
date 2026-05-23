# this code is to check the truthseeker dataset, which has many duplicates, 
# has no conflict between statement and target
# i.e, that none of the duplicates for statement have a target 
# contradicting the same statement's target in the sheet
# this is done before processing the dataset into a clean csv 
# to ensure no uncertain information is sent for testing.

import pandas as pd

df = pd.read_csv(r"..\datasets-raw\truthseeker\Truth_Seeker_Model_Dataset.csv")
df = df[['statement', 'target']]
df['target'] = df['target'].astype(str).str.lower().str.strip()

conflicts = df.groupby('statement')['target'].nunique()
conflicting_statements = conflicts[conflicts > 1].index

result = df[df['statement'].isin(conflicting_statements)]

print(result) #prints an empty dataframe (expected and desired result)