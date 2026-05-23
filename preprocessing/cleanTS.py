# this is to clean the truthseeker model dataset to remove duplicates of the statement being fed in
# as well as to normalize target in case of non-uniformity for true/false

import pandas as pd

df = pd.read_csv(r"..\datasets-raw\truthseeker\Truth_Seeker_Model_Dataset.csv")
df = df[['statement', 'target', 'author']]
df['statement'] = df['statement'].astype(str).str.strip()
df['target'] = df['target'].astype(str).str.lower().str.strip()

final_df = df.drop_duplicates(subset=['statement', 'target'])

final_df.to_csv(r"..\datasets-clean\clean_Truthseeker.csv", index=False)

print(final_df.head())

# print(final_df.head()) 
# prints the following:
#                                               statement target           author
# 0     End of eviction moratorium means millions of A...   true       D.L. Davis
# 488   The Trump administration worked to free 5,000 ...   true  Miriam Valverde
# 976   In Afghanistan, over 100 billion dollars spent...   true       D.L. Davis
# 1302  A photo shows two COVID-19 patients lying on t...   true      Amy Sherman
# 1316  Its been over 50 years since minimum (wage) an...   true    Madeline Heim
