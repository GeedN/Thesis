# the Liar dataset is split into 3 tsv files: train, test, and validate. 
# for the general results, the data is combined into one test file 
# as the models are pretrained so the train file would be unused
# however, for comparing with other studies, only the test set is used
# to ensure that the results remain relevant to other research
# additionally, the liar dataset contains more dimensions than true and false
# namely: ['true' 'false' 'half-true' 'pants-fire' 'barely-true' 'mostly-true']

import pandas as pd

def cleaner(filelist):
    dfs = []

    for file in filelist:
        df = pd.read_csv(file, sep="\t", header=None)

        # col 2 (index 1) = label/target, col 3 (index2)= statement, col 5 (index 4)= speaker/author
        df = df[[1, 2, 4]]
        df.columns = ["target", "statement", "author"]
        df["target"] = df["target"].astype(str).str.lower().str.strip()

        df = df[~df["target"].isin(["half-true", "barely-true", "mostly-true"])]
        df["target"] = df["target"].replace({"pants-fire": "false"})

        df["author"] = df["author"].astype(str).str.replace("-", " ", regex=False)

        dfs.append(df)
    
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.dropna(subset=["target"])
    final_df = final_df[["statement", "target", "author"]]

    return final_df


test_df = cleaner(["..\\datasets-raw\\liar\\test.tsv"])
test_df.to_csv("..\\datasets-clean\\clean_Liar_test.csv", index=False)

# if desired: (scrapped idea as liar test is better for comparability)
# combined_df = cleaner(["..\\datasets-raw\\liar\\train.tsv", "..\\datasets-raw\\liar\\test.tsv", "..\\datasets-raw\\liar\\valid.tsv"])
# combined_df.to_csv("..\\datasets-clean\\clean_Liar.csv", index=False)

# print(test_df.head())
# prints :
#                                            statement target                            author
# 0  Building a wall on the U.S.-Mexico border will...   true                        rick perry
# 1  Wisconsin is on pace to double the number of l...  false                 katrina shankland
# 2  Says John McCain has done nothing to help the ...  false                      donald trump
# 3  When asked by a reporter whether hes at the ce...  false  state democratic party wisconsin
# 4  Over the past five years the federal governmen...   true                   brendan doherty

# print(combined_df.head())
# prints:
#                                            statement target        author
# 0  Says the Annies List political group supports ...  false  dwayne bohac
# 1  Health care reform legislation is likely to ma...  false  blog posting
# 2  The Chicago Bears have had more starting quart...   true     robin vos
# 3  When Mitt Romney was governor of Massachusetts...  false   mitt romney
# 4  McCain opposed a requirement that the governme...   true  barack obama