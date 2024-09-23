import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

def calculate_missing_percentage(dataframe):
    # Determine the total number of elements in the DataFrame
    total_elements = np.prod(dataframe.shape)

    # Count the number of missing values in each column
    missing_values = dataframe.isna().sum()

    # Sum the total number of missing values
    total_missing = missing_values.sum()

    # Compute the percentage of missing values
    percentage_missing = (total_missing / total_elements) * 100

    # Print the result, rounded to two decimal places
    print(f"The dataset has {round(percentage_missing, 2)}% missing values.")


def check_missing_values(data):
    """Check for missing values in the dataset."""
    missing_values = data.isnull().sum()
    missing_percentages = 100 * data.isnull().sum() / len(data)
    column_data_types = data.dtypes
    missing_table = pd.concat([missing_values, missing_percentages, column_data_types], axis=1, keys=['Missing Values', '% of Total Values','Data type'])
    return missing_table.sort_values('% of Total Values', ascending=False).round(2)

def outlier_box_plots(data):
    for column in data:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[column])
        plt.title(f'Box plot of {column}')
        plt.show()