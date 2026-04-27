from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 

def train_model(X_train_sc , y_train):
    trained_model={}
    lr = LinearRegression()
    lr_param = { 
        'fit_intercept':[True ,False],
        'positive':[True ,False]
    }
    lr_grid = GridSearchCV(lr,lr_param , cv=5, scoring='neg_root_mean_squared_error' , n_jobs=-1)
    lr_grid.fit(X_train_sc ,y_train)
    trained_model['Linear Regression'] =lr_grid.best_estimator_


    rf = RandomForestRegressor(random_state=42)
    rf_param ={
        'n_estimators':[100,200],
        'max_depth':[10 ,20 ,None],
        'min_samples_split':[2,5],
    }
    rf_grid = GridSearchCV(rf , rf_param , cv=5 , scoring='neg_root_mean_squared_error' , n_jobs=-1)
    rf_grid.fit(X_train_sc ,y_train)
    trained_model['RandomForest Regression'] = rf_grid.best_estimator_


    xgb = XGBRegressor(random_state=42)
    xgb_param ={
        'n_estimators':[100,200],
        'max_depth':[3 ,5 ],
        'learning_rate':[0.01 ,0.1]
    }
    xgb_grid = GridSearchCV(xgb ,xgb_param ,cv=5 , scoring='neg_root_mean_squared_error' , n_jobs=-1) 
    xgb_grid.fit(X_train_sc ,y_train)
    trained_model['XGB Regressor'] = xgb_grid.best_estimator_

    print("\n All Models are Trained Sucessful !")
    return trained_model
