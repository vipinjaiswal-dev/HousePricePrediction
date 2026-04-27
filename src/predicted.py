import pandas as pd 

def prediction(best_model ,scaler=None):

    sample =pd.DataFrame([{
        'Area_sqr':1500,
        'Bedrooms':4 ,
        'Bathrooms':2 ,
        'Floors':4 ,
        'Age':5 ,
        'Location_score':7
    }])
    if scaler is not None:
        sample = scaler.transform(sample)
    prediction = best_model.predict(sample)

    print("\n Predicted House Price = ",round(prediction[0],2))
