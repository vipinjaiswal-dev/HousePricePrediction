from  src.data_loader import loaded_data
from src.precossing import precossing_data ,split_data ,scale_data
from src.model import train_model
from src.Evaluate import evaluate_data ,Actual_vs_prediction
from src.predicted import prediction
import joblib as jb
# data load 
df = loaded_data('data/house_price_datasets.csv')

# preprocessing data
X ,y = precossing_data(df)

# train And test data
X_train ,X_test ,y_train ,y_test = split_data(X ,y)

# split data
X_train_sc,X_test_sc ,scaler = scale_data(X_train ,X_test)

# Trained model
models = train_model(X_train_sc , y_train)

# Evaluate Data
best_model ,best_model_name , best_rmse ,results_df= evaluate_data(models ,X_test_sc ,y_test)

# Autcal Vs Prediction 
graphic = Actual_vs_prediction(models ,X_test_sc ,y_test)

# prediction 
predicted = prediction(best_model , scaler)

# Best Model 
jb.dump(best_model,'models/best_model.pkl')
jb.dump(scaler , 'models/scaler.pkl')