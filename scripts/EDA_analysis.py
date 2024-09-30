import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_promo_distribution(train, test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    train['Promo'].value_counts().plot(kind='pie', ax=ax1, autopct='%1.1f%%')
    test['Promo'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax1.set_title('Promo Distribution in Train Set')
    ax2.set_title('Promo Distribution in Test Set')
    plt.show()

def analyze_holiday_sales(data):
    # Ensure 'Date' is in datetime format
    if data['Date'].dtype != 'datetime64[ns]':
        data['Date'] = pd.to_datetime(data['Date'])
    
    # Analyze StateHoliday
    plt.figure(figsize=(12, 6))
    sns.barplot(x='StateHoliday', y='Sales', data=data)
    plt.title('Sales Distribution on State Holidays vs Non-Holidays')
    plt.xlabel('State Holiday (0: No, 1: Yes)')
    plt.ylabel('Sales')
    plt.show()

    # Analyze SchoolHoliday
    plt.figure(figsize=(12, 6))
    sns.barplot(x='SchoolHoliday', y='Sales', data=data)
    plt.title('Sales Distribution on School Holidays vs Non-Holidays')
    plt.xlabel('School Holiday (0: No, 1: Yes)')
    plt.ylabel('Sales')
    plt.show()

    # Analyze sales by day of week
    plt.figure(figsize=(12, 6))
    sns.barplot(x='DayOfWeek', y='Sales', data=data)
    plt.title('Sales Distribution by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Sales')
    plt.show()

def analyze_seasonal_sales(data):
    data['Month'] = data['Date'].dt.month
    monthly_sales = data.groupby('Month')['Sales'].mean()
    
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='bar')
    plt.title('Average Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.show()

# sales vs customer correlation 
def analyze_sales_customers_correlation(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Customers', y='Sales')
    plt.title('Sales vs Number of Customers')
    plt.show()

    correlation = data['Sales'].corr(data['Customers'])
    print(f"Correlation between Sales and Customers: {correlation}")

    # Create a correlation matrix
    corr_matrix = data[['Sales', 'Customers']].corr()

    # Heatmap for the correlation matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap: Sales vs Customers')
    plt.show()

def analyze_promo_effect(data):
    promo_effect = data.groupby('Promo')[['Sales', 'Customers']].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    promo_effect['Sales'].plot(kind='bar', ax=ax1)
    ax1.set_title('Average Sales with/without Promo')
    ax1.set_ylabel('Average Sales')
    
    promo_effect['Customers'].plot(kind='bar', ax=ax2)
    ax2.set_title('Average Customers with/without Promo')
    ax2.set_ylabel('Average Customers')
    
    plt.tight_layout()
    plt.show()


# Analyze promo effectiveness by store type
def enhance_promo_analysis(data):
    
    promo_effect_by_store_type = data.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack()
    promo_lift = (promo_effect_by_store_type[1] - promo_effect_by_store_type[0]) / promo_effect_by_store_type[0] * 100

    plt.figure(figsize=(10, 6))
    promo_lift.plot(kind='bar')
    plt.title('Promo Effectiveness by Store Type')
    plt.xlabel('Store Type')
    plt.ylabel('Sales Lift (%)')
    plt.show()

    # Suggest stores for promo deployment
    store_promo_effect = data.groupby('Store')[['Promo', 'Sales']].apply(lambda x: x[x['Promo'] == 1]['Sales'].mean() / x[x['Promo'] == 0]['Sales'].mean() - 1)
    top_stores_for_promo = store_promo_effect.nlargest(10)
    
    print("Top 10 stores where promos are most effective:")
    print(top_stores_for_promo)


def analyze_weekday_open_stores(data):
    # Identify stores open on all weekdays
    weekday_open_stores = data[(data['Open'] == 1) & (data['DayOfWeek'].isin([1, 2, 3, 4, 5]))].groupby('Store')['DayOfWeek'].nunique()
    always_open_stores = weekday_open_stores[weekday_open_stores == 5].index

    # Calculate average weekday and weekend sales for all stores
    weekday_sales = data[data['DayOfWeek'].isin([1, 2, 3, 4, 5])].groupby('Store')['Sales'].mean()
    weekend_sales = data[data['DayOfWeek'].isin([6, 7])].groupby('Store')['Sales'].mean()

    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Weekday Sales': weekday_sales,
        'Weekend Sales': weekend_sales,
        'Store Type': ['Always Open' if store in always_open_stores else 'Not Always Open' for store in weekday_sales.index]
    })

    # Melt the dataframe for easier plotting
    plot_data_melted = pd.melt(plot_data.reset_index(), id_vars=['Store', 'Store Type'], 
                               var_name='Day Type', value_name='Average Sales')

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Store Type', y='Average Sales', hue='Day Type', data=plot_data_melted)
    plt.title('Weekday vs Weekend Sales: Always Open Stores vs Others')
    plt.ylabel('Average Sales')
    plt.show()

    # Print summary statistics
    print("Summary Statistics:")
    print(plot_data.groupby('Store Type').agg({
        'Weekday Sales': ['mean', 'median'],
        'Weekend Sales': ['mean', 'median']
    }))

    # Analyze the difference in weekend vs weekday sales
    plot_data['Weekend_Weekday_Diff'] = plot_data['Weekend Sales'] - plot_data['Weekday Sales']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Store Type', y='Weekend_Weekday_Diff', data=plot_data)
    plt.title('Difference in Weekend vs Weekday Sales')
    plt.ylabel('Weekend Sales - Weekday Sales')
    plt.show()

    print("\nAverage difference in Weekend vs Weekday sales:")
    print(plot_data.groupby('Store Type')['Weekend_Weekday_Diff'].mean())



def analyze_assortment_effect(data):
    assortment_effect = data.groupby('Assortment')['Sales'].mean()
    
    plt.figure(figsize=(10, 6))
    assortment_effect.plot(kind='bar')
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.show()

def analyze_competition_distance(data):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='CompetitionDistance', y='Sales')
    plt.title('Sales vs Competition Distance')
    plt.xlabel('Competition Distance (meters)')
    plt.ylabel('Sales')
    plt.show()
    correlation = data['Sales'].corr(data['CompetitionDistance'])
    print(f'Sales vs Competition Distance (Correlation: {correlation:.2f})')

def analyze_new_competitors(data):
    data['CompetitorAge'] = data['Date'].dt.year - data['CompetitionOpenSinceYear']
    data['CompetitorAge'] = data['CompetitorAge'].clip(lower=0)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='CompetitorAge', y='Sales')
    plt.title('Average Sales vs Competitor Age')
    plt.xlabel('Years Since Competitor Opened')
    plt.ylabel('Average Sales')
    plt.show()

def analyze_competitor_effect(data):
    # Filter stores that initially had no CompetitionDistance but later have values
    competitors_changed = data[(data['CompetitionDistance'].isna()) & 
                               (data['CompetitionOpenSinceYear'].notna())]

    print(f"Total stores affected by new competitor opening: {competitors_changed['Store'].nunique()}")

    # Fill missing CompetitionDistance with 0 for better handling
    data['CompetitionDistance'] = data['CompetitionDistance'].fillna(0)

    # Analyze average sales before and after competition opens
    # Calculate the years after competition opened
    data['CompetitorAge'] = data['Date'].dt.year - data['CompetitionOpenSinceYear']
    data['CompetitorAge'] = data['CompetitorAge'].fillna(-1).clip(lower=-1) # -1 means no competitor
    
    # Plot sales against the age of competitors
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='CompetitorAge', y='Sales', hue='StoreType')
    plt.title('Average Sales vs Competitor Age (for stores with new competitors)')
    plt.xlabel('Years since Competitor Opened')
    plt.ylabel('Average Sales')
    plt.show()

    # Analyze the stores before and after competition opening (filtering for stores with competitor data)
    competitor_opening_stores = data[data['CompetitionOpenSinceYear'].notna()]
    
    # Compare sales for stores with new competitors before and after the competitor opens
    competitor_opening_stores['BeforeCompetitor'] = (data['Date'].dt.year < data['CompetitionOpenSinceYear'])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=competitor_opening_stores, x='BeforeCompetitor', y='Sales')
    plt.title('Sales Before vs After Competitor Opening')
    plt.xticks([0, 1], ['After Competitor Opened', 'Before Competitor Opened'])
    plt.xlabel('Competitor Presence')
    plt.ylabel('Sales')
    plt.show()

    # Sales trend as competitor age grows
    competitor_sales_trend = competitor_opening_stores.groupby('CompetitorAge')['Sales'].mean()
    plt.figure(figsize=(12, 6))
    competitor_sales_trend.plot(kind='line')
    plt.title('Sales Trend as Competitor Gets Older')
    plt.xlabel('Competitor Age (Years)')
    plt.ylabel('Average Sales')
    plt.show()

    # Correlation between competition distance and sales after the competitor opened
    correlation = data[data['CompetitionDistance'] > 0]['Sales'].corr(data['CompetitionDistance'])
    print(f"Correlation between Competition Distance and Sales: {correlation:.2f}")

def analyze_store_hours(data):
    # Group by DayOfWeek and calculate average sales
    daily_sales = data.groupby('DayOfWeek')['Sales'].mean().reset_index()
    
    # Create a bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='DayOfWeek', y='Sales', data=daily_sales)
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Sales')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()

    # Analyze open/closed patterns
    open_stores = data.groupby('DayOfWeek')['Open'].mean()
    print("Proportion of stores open by day of week:")
    print(open_stores)

    # Analyze sales for open stores
    open_sales = data[data['Open'] == 1].groupby('DayOfWeek')['Sales'].mean()
    print("\nAverage sales for open stores by day of week:")
    print(open_sales)