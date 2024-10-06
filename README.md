# Time Series Analysis of Air Passenger Data

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Steps Taken](#steps-taken)
- [Recommendations](#recommendations)

### Project Overview

This project focuses on analyzing the monthly air passenger data from January 1949 to December 1960. The goal is to understand trends, seasonality, and fluctuations in the data and to build predictive models using linear regression. The project also includes log transformation to stabilize seasonal components for better predictive modeling.

### Dataset

The dataset used in this project is titled as airpassengers.csv. It includes the following columns:

- Year-Month: Date of observation (monthly data)
- Passengers: Number of passengers (in thousands)

### Project Workflow

1. Importing required libraries:
   ``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')
```
``` SQL
SELECT *
FROM hr_data;
```
3. Data Import and Preprocessing:
- Import the CSV file and ensure that the Year-Month column is parsed as a date type.

- Basic data exploration and checking for any missing values or inconsistencies.

3. 

