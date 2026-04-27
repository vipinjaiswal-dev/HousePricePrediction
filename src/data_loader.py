import pandas as pd 

def loaded_data (path):
    df=pd.read_csv(path)
    return df 