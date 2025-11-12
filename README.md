# Used Car Price Prediction Using Machine Learning and Streamlit

## Domain
Automotive Industry * Data Science * Machine Learning

## Introduction
This project focuses on improving the customer experience and optimizing the pricing process for used cars through a machine learning model.
Using historical car price data, the model predicts accurate selling prices based on multiple features such as the car’s make, model, year, fuel type, transmission type, kilometers driven, and ownership type.

The goal of this project is to develop an accurate price prediction model and integrate it into an interactive Streamlit web application.
This allows users to easily estimate the market value of used cars in real time.

## Skills Takeaway From This Project
* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Machine Learning Model Development
* Price Prediction Techniques
* Model Evaluation and Optimization
* Model Deployment using Pickle
* Streamlit Application Development
* Documentation and Reporting

## Technologies Used
* Category	Tools / Frameworks
* Programming Language	Python
* IDE / Environment	Jupyter Notebook, VS Code
* Web Framework	Streamlit
* Machine Learning Framework	Scikit-learn
* Visualization	Matplotlib, Seaborn
* Data Handling	Pandas, NumPy
* Deployment	Pickle Serialization

## Packages and Libraries
* Purpose	Libraries
* Data Handling	pandas, numpy
* Visualization	matplotlib, seaborn
* Machine Learning	scikit-learn, xgboost
* Model Saving / Loading	pickle, joblib
* Web Deployment	streamlit
* Hyperparameter Tuning	GridSearchCV, RandomizedSearchCV

## Project Features
* Data Cleaning and Preprocessing – Handling missing values, encoding categorical data, and feature scaling.
* Exploratory Data Analysis – Understanding correlations between car features and selling prices.
* Model Building – Trained regression models including Linear Regression, Random Forest, and XGBoost.
* Hyperparameter Optimization – Used Grid Search to find the best model parameters.
* Model Deployment – Saved the model, encoder, and scaler using pickle for real-time use.
* Streamlit Integration – Built a user interface for real-time predictions.
* Visualization – Displayed feature importance and model performance metrics.
* User-Friendly Interface – Simple and clean UI for user input and output.
  
## Results
The Random Forest Regressor achieved the best performance with the highest R² score and the lowest MAE/MSE, making it the final model for deployment.
Hyperparameter tuning using Grid Search helped identify the optimal parameters such as n_estimators and max_depth to enhance the model’s performance.

* Metric	Score
* R² Score	0.93
* MAE	1.05
* RMSE	1.21\

## Streamlit Application Development
The predictive model was deployed using the Streamlit web framework.
This application allows users to input various car details and instantly receive an estimated market price.
The Streamlit app provides a fast, accurate, and user-friendly experience, making it a useful tool for car dealerships and customers.

## Project Outcome
* Developed an end-to-end machine learning model for car price prediction.
* Integrated the trained model into an interactive Streamlit application.
* Demonstrated strong model performance with reliable predictions.
* Showcased practical use of machine learning in real-world automotive pricing.
