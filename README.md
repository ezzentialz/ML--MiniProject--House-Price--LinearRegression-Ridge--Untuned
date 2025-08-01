# ML--MiniProject--House-Price--LinearRegression-Ridge--Untuned
For Beginner learning with simple Linear Regression (Ridge) model - untuned, using OnehotEncodoing with pd.get_dummies method, using log1p to scale large digits of numbers, and simple evaluate model with metrics:MAE/MSE/RMSE/R2 


I will start with using Simple Linear Regression Model - for basics beginers ML. Please noted that my English is not good, but I will try my best to explain it easiest way for you guys to follow (> I'm for Thailand).

If you guys have any comments, recommednations, or suggestions, I'm very happy to hear from you.

(* This is not the best solution model for this House Price Prediction and this practice does not use Hyperparameter tuning - *just a simple data cleaning-fill missing values, simple onehotencoding(using pd.get_dummies), simple train-fit-predict model, and simple evaluate model with metrics:MAE/MSE/RMSE/R2 for beginers practice and learn from basics**)


# Boston Housing Price Prediction

This project aims to predict the median house prices in Boston using a Linear Regression and Ridge Regression model. The analysis is based on the famous Boston Housing dataset.

## Analysis Steps
1.  **Data Preprocessing:** Handled missing values and outliers to ensure data quality.
2.  **Exploratory Data Analysis (EDA):** Visualized the distribution of features and their correlation with the target variable.
3.  **Model Training:** Trained models:
    * **Ridge Regression**
4.  **Model Evaluation:** Evauation of the performance models using MAE, RME, RMSE and R2 scores.

## Results
- The Ridge Regression model performed slightly better than the standard Linear Regression model, indicating its effectiveness in handling multicollinearity in the dataset.

## Tools
- Python
- Pandas
- NumPy
- Scikit-learn
