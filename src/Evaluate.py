from sklearn.metrics import mean_absolute_error ,r2_score ,mean_squared_error
import numpy as np 
import pandas as pd 
from tabulate import tabulate
import matplotlib.pyplot as plt 
import seaborn as sns
def evaluate_data(models , X_test_sc , y_test):
    results=[]
    
    best_rmse =float('inf')
    best_model = None
    best_model_name =""

    for name ,model in models.items():
        y_pred = model.predict(X_test_sc)

        mae = mean_absolute_error (y_test , y_pred)
        rmse =np.sqrt (mean_squared_error (y_test ,y_pred))
        r2 = r2_score(y_test ,y_pred)

        results.append({
            'Model Name ':name ,
            'MAE':round(mae,1),
            'RMSE':round(rmse ,1),
            'R2': round(r2 ,4)  
        })

        # Best model Selectn
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)
    print(tabulate(results_df , headers='keys' ,tablefmt='grid'))
    print(f"\n Best Model = {best_model_name}")
    print(f"Best RMSE = {round(best_rmse ,1)}")
    return best_model ,best_model_name ,best_rmse , results_df

def Actual_vs_prediction(models , X_test_sc , y_test):

    for name , model in models.items():
        y_pred = model.predict(X_test_sc)

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=y_test ,y=y_pred)

        plt.plot(
            [min(y_test), max(y_test)],
            [min(y_test) , max(y_test)],
            color='red',
            linestyle ='--'
        )

        plt.title(f"{name} ")
        plt.xlabel("Actual Price")
        plt.ylabel("predicted Price")
        plt.grid(True)
        plt.show()
 
   
