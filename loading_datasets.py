import pandas as pd

SMALL_DATASETS = False # to speed up testing the code during development

def load_truthseeker():
    df = pd.read_csv("datasets-clean/clean_truthseeker.csv", sep = ",", index_col = 0)
    
    df = df.reset_index()
    
    if SMALL_DATASETS:
        df = df.sample(frac=50/len(df), random_state=42).reset_index(drop=True)
        assert len(df) == 50
    
    return df

def load_liar():
    df = pd.read_csv("datasets-clean/clean_liar_test.csv", sep = ",", index_col = 0)
    
    df = df[df["author"].notna()] # there are two rows without an author, which are filtered out 
    
    df = df.reset_index()
    
    df = df[~df["statement"].str.contains(".json", na=False)] # exclude a couple malformed rows
    
    if SMALL_DATASETS:
        df = df.sample(frac=50/len(df), random_state=42).reset_index(drop=True)
        assert len(df) == 50
    
    return df
    
if __name__ == "__main__":
    df = load_truthseeker()
    print(df.columns)
    print(df.head())
    
    df = load_liar()
    print(df.columns)
    print(df.head())