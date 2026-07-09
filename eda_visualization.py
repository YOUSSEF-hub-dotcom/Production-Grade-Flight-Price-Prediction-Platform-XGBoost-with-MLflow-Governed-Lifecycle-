import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger("EDA")

def run_eda(df):
    logger.info("=========== EDA & Visualization ==========")

    # ----------------------------------------------------
    # 1. Target Variable Distribution (Price)
    # ----------------------------------------------------
    logger.info("What is the distribution of airfare prices?")
    print(df['Price'].describe())

    plt.figure(figsize=(8,5))
    sns.histplot(df['Price'], bins=50, kde=True)
    plt.title("Distribution of Flight Ticket Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 2. Airline vs Price Analysis (Statistical Summary & Spread)
    # ----------------------------------------------------
    logger.info("Does the ticket price vary depending on the airline?")
    ticket_price_per_Airline = df.groupby('Airline')['Price'].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
    print(ticket_price_per_Airline)

    plt.figure(figsize=(10,5))
    sns.boxplot(x='Airline', y='Price', data=df, color='blue')
    plt.title("Ticket Price Distribution by Airline")
    plt.xticks(rotation=45)
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 3. Airline Ranking by Average Price
    # ----------------------------------------------------
    logger.info("Are there airlines whose prices are noticeably higher than others?")
    Avg_Price_Airline = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
    print(Avg_Price_Airline)

    plt.figure(figsize=(10,5))
    Avg_Price_Airline.plot(kind='barh')
    plt.title("Average Ticket Price per Airline")
    plt.xlabel("Average Price")
    plt.ylabel("Airline")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 4. Within-Airline Price Variance (Violin Plots for Density)
    # ----------------------------------------------------
    logger.info("Is the price difference within the same airline significant?")
    STD_Price_Airline = df.groupby('Airline')['Price'].std().sort_values(ascending=False)
    print(STD_Price_Airline)

    plt.figure(figsize=(10,5))
    sns.violinplot(x='Airline', y='Price', data=df)
    plt.title("Price Variability Within Each Airline")
    plt.xticks(rotation=45)
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 5. Source City Impact on Price
    # ----------------------------------------------------
    logger.info("Does the departure city(Source) affect the ticket price?")
    Avg_Price_Source = df.groupby('Source')['Price'].mean().sort_values(ascending=False)
    print(Avg_Price_Source)

    plt.figure(figsize=(8,5))
    Avg_Price_Source.plot(kind='bar', color='blue')
    plt.title("Ticket Price by Source City")
    plt.ylabel("Average Price")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 6. Destination City Impact on Price
    # ----------------------------------------------------
    logger.info("Does the arrival city(Destination) affect the ticket price?")
    Avg_Price_Destination = df.groupby('Destination')['Price'].mean().sort_values(ascending=False)
    print(Avg_Price_Destination)

    plt.figure(figsize=(8,5))
    sns.boxplot(x='Destination', y='Price', data=df, color='green')
    plt.title("Ticket Price by Destination City")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 7. Number of Stops Impact on Price
    # ----------------------------------------------------
    logger.info("Does the ticket price vary depending on the number of stops?")
    TotalStops_Price_of_Ticket = df.groupby('Total_Stops')['Price'].agg(['mean', 'median', 'std'])
    print(TotalStops_Price_of_Ticket)

    plt.figure(figsize=(8,5))
    sns.barplot(x='Total_Stops', y='Price', data=df, color='Maroon')
    plt.title("Ticket Price by Number of Stops")
    plt.xlabel("Total Stops")
    plt.ylabel("Price")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 8. Departure Hour Hourly Trend
    # ----------------------------------------------------
    logger.info("Does the departure time(Dep_hour) affect the ticket price?")
    Dep_hour_affectOnPrice = df.groupby('Dep_hour')['Price'].mean()
    print(Dep_hour_affectOnPrice.head()) # Fixed: Only head the print, keep full data for lineplot

    plt.figure(figsize=(10,5))
    sns.lineplot(x='Dep_hour', y='Price', data=df, estimator='mean', color='coral')
    plt.title("Average Price vs Departure Hour")
    plt.grid(True)
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 9. Time Session (Morning vs Night, etc.) Impact
    # ----------------------------------------------------
    logger.info("Are morning flights more expensive than night flights?")
    Deep_Session_per_price = df.groupby('Dep_Session')['Price'].mean().sort_values(ascending=False)
    print(Deep_Session_per_price)

    plt.figure(figsize=(8,5))
    sns.barplot(x='Dep_Session', y='Price', color='teal', data=df)
    plt.title("Average Price by Departure Session")
    plt.xticks(rotation=15)
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 10. Arrival Hour Hourly Trend
    # ----------------------------------------------------
    logger.info("Does the arrival time affect the price?")
    Arrival_time_affectOnPrice = df.groupby('Arrival_hour')['Price'].mean()
    print(Arrival_time_affectOnPrice.head()) # Fixed: Only head the print

    plt.figure(figsize=(10,5))
    sns.lineplot(x='Arrival_hour', y='Price', data=df, estimator='mean', color='purple')
    plt.title("Average Price vs Arrival Hour")
    plt.grid(True)
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 11. Flight Duration Correlation with Price
    # ----------------------------------------------------
    logger.info('Does the duration of the trip affect the ticket price?')
    print(df[['Duration_mins','Price']].corr())

    plt.figure(figsize=(8,5))
    sns.scatterplot(x='Duration_mins', y='Price', data=df, alpha=0.5)
    plt.title("Price vs Flight Duration")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 12. Linearity Check (Pearson Correlation & Regression Line)
    # ----------------------------------------------------
    logger.info('Is the relationship between flight duration and price linear?')
    print(df[['Duration_mins','Price']].corr(method='pearson'))

    plt.figure(figsize=(8,5))
    sns.regplot(x='Duration_mins', y='Price', data=df, scatter_kws={'alpha':0.3})
    plt.title("Linear Relationship Between Duration and Price")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 13. Categorical Flight Duration Impact (Long vs Short)
    # ----------------------------------------------------
    logger.info('Are long flights always more expensive than short flights?')
    print(df.groupby('Is_Long_Flight')['Price'].mean())

    plt.figure(figsize=(6,5))
    sns.boxplot(x='Is_Long_Flight', y='Price', data=df, color='purple')
    plt.title("Price Comparison: Long vs Short Flights")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 14. Day of the Week Impact on Price
    # ----------------------------------------------------
    logger.info('Does the ticket price vary depending on the day of the trip?')
    print(df.groupby('Day_of_Week')['Price'].mean())

    plt.figure(figsize=(8,5))
    sns.barplot(x='Day_of_Week', y='Price', data=df)
    plt.title("Average Price by Day of Week")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 15. Weekend vs Weekday Feature Verification
    # ----------------------------------------------------
    logger.info('Are weekend trips more expensive than other days?')
    print(df.groupby('is_weekend')['Price'].mean())

    plt.figure(figsize=(6,5))
    sns.boxplot(x='is_weekend', y='Price', data=df)
    plt.title("Weekend vs Weekday Prices")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 16. Monthly Trends (Seasonality Check)
    # ----------------------------------------------------
    logger.info('Does the ticket price vary from month to month?')
    print(df.groupby('Month_of_Journey')['Price'].mean())

    plt.figure(figsize=(10,5))
    sns.lineplot(x='Month_of_Journey', y='Price', data=df, estimator='mean')
    plt.grid(True)
    plt.title("Average Price by Month")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 17. Peak Season Binary Feature Verification
    # ----------------------------------------------------
    logger.info('Does the peak season lead to higher prices?')
    print(df.groupby('is_peak_season')['Price'].mean())

    plt.figure(figsize=(6,5))
    sns.boxplot(x='is_peak_season', y='Price', data=df)
    plt.title("Peak Season vs Non-Peak Season Prices")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 18. Interaction: Airline vs Total Stops Heatmap
    # ----------------------------------------------------
    logger.info("Does the airline's impact on price vary depending on the number of stops?")
    Airline_Price_Total_Stops = pd.pivot_table(df, values='Price', index='Airline', columns='Total_Stops', aggfunc='mean')
    print(Airline_Price_Total_Stops)

    plt.figure(figsize=(10,6))
    sns.heatmap(Airline_Price_Total_Stops, annot=True, fmt=".0f", cmap="coolwarm")
    plt.title("Airline vs Stops (Average Price)")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 19. Interaction: Departure Hour vs Day of Week Heatmap
    # ----------------------------------------------------
    logger.info('Does the effect of the departure time differ between days of the week?')
    Dep_hour_Day_of_Week_onPrice = pd.pivot_table(df, values='Price', index='Dep_hour', columns='Day_of_Week', aggfunc='mean')
    print(Dep_hour_Day_of_Week_onPrice.head()) # Fixed: Only head the print, plot full data

    plt.figure(figsize=(10,6))
    sns.heatmap(Dep_hour_Day_of_Week_onPrice, cmap='viridis')
    plt.title("Departure Hour vs Day of Week (Price)")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 20. Interaction: Flight Route (Path) vs Peak Season Heatmap
    # ----------------------------------------------------
    logger.info('Does the price of the route change during peak season compared to regular seasons?')
    Path_is_Peak_Session_onPrice = pd.pivot_table(df, values='Price', index='Path', columns='is_peak_season', aggfunc='mean')
    print(Path_is_Peak_Session_onPrice.head()) # Fixed: Only head the print

    plt.figure(figsize=(10,6))
    sns.heatmap(Path_is_Peak_Session_onPrice, cmap='coolwarm')
    plt.title("Path Price in Peak vs Non-Peak Season")
    plt.show()
    plt.close()

    # ----------------------------------------------------
    # 21. Multi-dimensional: Duration, Stops, Airline and Price
    # ----------------------------------------------------
    logger.info('What is the impact of (airline + number of stops + flight duration) on the price?')
    impactOf_Air_Total_Dura_onPrice = df.groupby(['Airline','Total_Stops'])[['Duration_mins','Price']].mean()
    print(impactOf_Air_Total_Dura_onPrice.head())

    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=df,
        x='Duration_mins',
        y='Price',
        hue='Total_Stops',
        style='Airline',
        alpha=0.6
    )
    plt.title("Combined Effect of Airline, Stops & Duration on Price")
    plt.grid(True)
    plt.show()
    plt.close()
