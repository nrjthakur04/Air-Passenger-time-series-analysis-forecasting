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

1. Importing required libraries:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

2. Data Import and Preprocessing:
- Import the CSV file and ensure that the Year-Month column is parsed as a date type.
```python
# Read the data
df=pd.read_csv('airpassengers.csv')
# Check data types
df.dtypes
#Data type of year=month column is of type object so, we are providing inputs to tell pandas that we're 
# trying to work with the time series and make it to time series using-
df=pd.read_csv('airpassengers.csv',parse_dates=['Year-Month'])
# Check data types
df.dtypes
df.head()
# To conviniently extract passenger values for a specific time period it is recommended that we
# Make the time series reference(column) as index-
df=pd.read_csv('airpassengers.csv',parse_dates=['Year-Month'],index_col='Year-Month')
df.head()
# Ṇow we can conveniently do slicing (we can obtain data for a pecific time period)--
df['1955-04':'1956-06']
# We can also check values for a specific time point using-
df.loc['1960-04']
# To understand our time series data in a better way, we'll first plot the time series so as to view the data fluctuations
df.plot()
plt.show()
# To see the image clearly and more asthetically we'll increse the figure size
from pylab import rcParams
rcParams['figure.figsize'] = 12,8
df.plot(color='green',linewidth=3)
plt.title('Passengers over Year-Month',fontweight='bold')
plt.ylabel('Passengers',fontweight='bold')
plt.show()
# As we can see the trend is countinously increasing (seanonal component varies wrt each other in different time series) and the seasonality is not constant.
# We know that if seasonality is not constant and changes over trend then we use multiplicative model to decompose the time series.




- Basic data exploration and checking for any missing values or inconsistencies.

3. 

