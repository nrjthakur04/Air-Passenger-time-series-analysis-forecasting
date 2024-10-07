# Time Series Analysis of Air Passenger Data

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Recommendations](#recommendations)

### Project Overview

This project focuses on analyzing the monthly air passenger data from January 1949 to December 1960. The goal is to understand trends, seasonality, and fluctuations in the data and to build predictive models using linear regression. The project also includes log transformation to stabilize seasonal components for better predictive modeling.

### Dataset

The dataset used in this project is titled as airpassengers.csv. It includes the following columns:

- Year-Month: Date of observation (monthly data)
- Passengers: Number of passengers (in thousands)

### Project Workflow

### 1. Importing required libraries:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')
```

### 2. Data Import and Preprocessing:
- Import the CSV file and ensure that the Year-Month column is parsed as a date type.
```python
# Read the data
df=pd.read_csv('airpassengers.csv')
```
```python
# Check data types
df.dtypes
```
![Screenshot (575)](https://github.com/user-attachments/assets/842d4e10-83fe-43dd-8d5e-4db3c27780d3)

- Data type of year=month column is of type object so, we are providing inputs to tell pandas that we're trying to work with the time series and make it to time series using-
```python
df=pd.read_csv('airpassengers.csv',parse_dates=['Year-Month'])
```
```python
# Check data types
df.dtypes
```
![Screenshot (576)](https://github.com/user-attachments/assets/c668ba21-1923-43d6-b468-a0f32702fb42)

```python
# See data of first 5 rows
df.head()
```
![Screenshot (577)](https://github.com/user-attachments/assets/ac051230-a325-42ae-a7cb-d76b2eb62d5e)

- To conviniently extract passenger values for a specific time period it is recommended that we make the time series reference(column) as index-
```python
df=pd.read_csv('airpassengers.csv',parse_dates=['Year Month'],index_col='Year-Month')
df.head()
```
- Ṇow we can conveniently do slicing (we can obtain data for a pecific time period)--
```python
df['1955-04':'1956-06']
```
![Screenshot (578)](https://github.com/user-attachments/assets/e5cc946c-18a0-4eee-b9b3-862595514995)

- We can also check values for a specific time point using-
```python
df.loc['1960-04']
```
![Screenshot (579)](https://github.com/user-attachments/assets/69cdd6f5-94be-44cc-9bb0-169f5a557031)

### 3. Time Series Plot:
- To understand our time series data in a better way, we'll first visualize the original time series to observe patterns in passenger numbers over time.
```python
df.plot()
plt.show()
```
![Screenshot (580)](https://github.com/user-attachments/assets/64e8cf25-8a86-4a68-b914-e6beca58bacb)

- To see the image clearly and more asthetically we'll increase the figure size
```python
from pylab import rcParams
rcParams['figure.figsize'] = 12,8
df.plot(color='green',linewidth=3)
plt.title('Passengers over Year-Month',fontweight='bold')
plt.ylabel('Passengers',fontweight='bold')
plt.show()
```
![Screenshot (581)](https://github.com/user-attachments/assets/0aa77d5f-0e92-4e22-931b-52211c904cac)

### 4. Time Series Decomposition:
- As we can see the trend is countinously increasing (seasonal component varies wrt each other in different time series) and the seasonality is not constant.
- Use the seasonal_decompose() function to break down the time series into its components: trend, seasonality, and residuals.
- We know that if seasonality is not constant and changes wrt trend then we'll use multiplicative model to decompose the time series, and then apply a log transformation to convert it into an additive model.
```python
df_mul_decompose=seasonal_decompose(df,model='multiplicative')
df_mul_decompose.plot()
plt.show()
```
![Screenshot (582)](https://github.com/user-attachments/assets/20cdc786-4622-4aec-91ac-b5af9b219acf)
### 5. Log Transformation:
- Log transformation of passenger numbers to stabilize the seasonal components and make the series suitable for additive models.
### - logy=logtrend+logseasonality+logresidual(irregularity)
- By doing so we can remove the change in seasonality, so the seasonality becomes constant and we can represent this multiplicative model as additive model
```python
df_log=df.copy()
```
```python
df_log['Passengers']=np.log(df)
```
```python
df_log.Passengers
```
![Screenshot (583)](https://github.com/user-attachments/assets/d8b12e3d-0220-4165-9a3e-b379baf8be76)

- Visualize the log transformed series
```python
from pylab import rcParams
rcParams['figure.figsize'] = 12,8
df_log.plot(color='green',linewidth=3)
plt.title('Passengers over Year-Month',fontweight='bold')
plt.xlabel('Year-Month',fontweight='bold')
plt.ylabel('Passengers',fontweight='bold')
plt.show()
```
![Screenshot (584)](https://github.com/user-attachments/assets/b02ffdc8-1a54-4c40-a184-dbc209f979db)

" In above figure, we can see that the variation in seasonal pattern has reduced(If we compare the seasonal pattern in last 4 they are same or constant)"

### 6. Model Comparison:
- Compare the original time series with the log-transformed series to identify the differences in seasonality and trend stability.
```python
plt.subplot(2,1,1)
plt.title('Original Time Series',fontweight='bold')
plt.plot(df,color='red',linewidth=3)

plt.subplot(2,1,2)
plt.title('Log Transformed Time Series',fontweight='bold')
plt.plot(df_log,color='green',linewidth=3)
plt.tight_layout()
```
![Screenshot (585)](https://github.com/user-attachments/assets/277f9edf-d2de-4b40-9e21-a77d486aa26c)

" In Original time series we can see the upward trend for seasonality, while in Log Transformed Time Series seasonality is constant"

```python
df.Passengers
```
![Screenshot (586)](https://github.com/user-attachments/assets/34c030cb-02eb-430e-95cc-8cb6949678ed)

## Linear Regression for Future Prediction:
- Linear Regression (y=a+bx)
- Use linear regression to model and predict future air passenger numbers based on the log-transformed data
- Visualize the results and evaluate the model's performance.
```python
# Importing libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```
### Data Preparation:
- We first log-transform the passenger data to stabilize variance.
```python
# Prepare the data
df_log = pd.read_csv('airpassengers.csv', parse_dates=['Year-Month'])
df_log['Passengers'] = np.log(df_log['Passengers'])
```
- The 'Year-Month' column is converted into a numerical format (ordinal values) to be used as the independent variable in linear regression.
```python
# Convert 'Year-Month' to ordinal (numerical format)
df_log['Year-Month'] = pd.to_datetime(df_log['Year-Month'])
df_log['Time'] = df_log['Year-Month'].map(pd.Timestamp.toordinal)
```
- Define the feature and target variable
```python
X = df_log[['Time']]  # Time as the independent variable
y = df_log['Passengers']  # Log-transformed passengers as the dependent variable.
```
### Model Training:
- We split the data into training and test sets, fitting the linear regression model to the training data.
```python
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```
```python
# Fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
```
### Future Prediction:
- The model is used to predict passenger counts for the next 12 months.
```python
# Make predictions for training and test data
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
```
```python
# Predict future values (let's predict for the next 12 months)
future_periods = 12
last_date = df_log['Year-Month'].max()
future_dates = pd.date_range(last_date, periods=future_periods+1, freq='MS')[1:]  # Monthly start
```
```python
# Convert future dates to ordinal format
future_ordinals = future_dates.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
```
```python
# Make predictions for future
future_predictions = model.predict(future_ordinals)
```
### Inverse Log Transformation:
- After predicting, we apply the exponential function (np.exp) to the predicted log-transformed values to convert them back to the original passenger scale.
```python
# Inverse log transformation to bring back to original scale
future_passengers = np.exp(future_predictions)
```
### Visualization:
- Below graph shows the original data, the fitted values for both training and test data, and the future predictions.
```python
# Visualize and plot original log-transformed data
plt.figure(figsize=(12,8))
plt.plot(df_log['Year-Month'], np.exp(y), label='Original Data', color='green')
```
![Screenshot (587)](https://github.com/user-attachments/assets/72e62ac7-3cf3-4e14-966a-b102fd58f74e)

```python
# Plot predicted values for training and test sets using their respective indices
plt.plot(df_log['Year-Month'].iloc[:len(y_pred_train)], np.exp(y_pred_train), label='Fitted Training Data', color='blue')
plt.plot(df_log['Year-Month'].iloc[-len(y_pred_test):], np.exp(y_pred_test), label='Predicted Test Data', color='orange')
```
![Screenshot (588)](https://github.com/user-attachments/assets/75f7b271-62ba-4031-9b20-368b47a0f075)

```python
# Plot future predictions
plt.plot(future_dates, future_passengers, label='Future Predictions', color='red', linestyle='dashed')

plt.title('Linear Regression for Time Series Prediction')
plt.xlabel('Year-Month')
plt.ylabel('Passengers')
plt.legend()
plt.grid(True)
plt.show()
```
![Screenshot (589)](https://github.com/user-attachments/assets/53ba9586-db8a-4f5c-a0d3-c38b48a0409d)

