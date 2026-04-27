
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
def precossing_data(df):
    print(f"\nMissing value { df.isnull().sum()}\n")
    df.drop_duplicates(inplace=True)

    X = df.drop('Price' , axis=1)
    y = df['Price']
    return X , y 

# Train and Test data 
def split_data (X ,y):
    X_train ,X_test ,y_train , y_test = train_test_split(X ,y ,test_size=0.2,random_state=42)
    print("\n split Data is successful !")
    return X_train ,X_test ,y_train ,y_test 

def scale_data(X_train ,X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    print("\n Precossing is successful !")
    return X_train_sc,X_test_sc ,scaler