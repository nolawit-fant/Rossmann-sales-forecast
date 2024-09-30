# Rossmann Pharmaceuticals Sales Forecasting
This project aims to forecast sales for Rossmann Pharmaceuticals stores across several cities, leveraging machine learning and deep learning techniques. The predictions are intended to assist the finance team in planning six weeks ahead.

## Project Overview
The project involves building an end-to-end solution for predicting store sales by exploring customer purchasing behavior and employing both machine learning and deep learning models. The predictions are served through a REST API for real-time use.

## Table of Contents
Business Need  
Data and Features  
Project Tasks  
## Business Need
Rossmann Pharmaceuticals requires accurate sales forecasts to optimize operations and financial planning. This project addresses the need for predicting daily sales in various stores up to six weeks in advance.

## Data and Features
The dataset includes various fields such as:  

Store: Unique store ID  
Sales: Daily turnover  
Customers: Number of customers  
Open: Store open indicator  
StateHoliday, SchoolHoliday: Holiday indicators  
StoreType, Assortment: Store characteristics  
CompetitionDistance: Distance to nearest competitor  
Promo, Promo2: Promotional details  
## Project Tasks
Exploration of Customer Purchasing Behavior: Analyze how factors like promotions and competitor openings affect sales.  
Prediction of Store Sales: Build regression models using scikit-learn and deep learning models with LSTM.  
Model Serving: Create a REST API to serve predictions using Flask or FastAPI.
