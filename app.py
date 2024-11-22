import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Add Custom CSS for Frontend
st.markdown("""
    <style>
        body { font-family: 'Poppins', sans-serif; background-color: #f8f8f8; }
        h1, h2, h3 { color: #5a5a5a; }
        .sidebar .sidebar-content { background: linear-gradient(120deg, #A49CF4, #F899CE); color: white; }
        .stButton>button { color: white; background: #F899CE; border-radius: 5px; border: none; }
    </style>
""", unsafe_allow_html=True)

# Load datasets
df_behavioral = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Behavioral_Dataset.csv')
df_customer = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Customer_Dataset.csv')
df_item = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Transaction_Items_Dataset.csv')
df_transaction = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Transaction_Dataset.csv')
df_inventory = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Inventory_Dataset.csv')
df_campaign = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Campaign_Dataset.csv')

# Sidebar Navigation
st.sidebar.title("Dynamic BI for Retail")
page = st.sidebar.radio(
    "Navigate",
    options=["Browsing Pattern", "Seasonal & Geographic", "Price Sensitivity", "Funnel Analysis", "Inventory Analysis"]
)

# Title
st.title("Retail Data Analysis Dashboard")

# Function Definitions for Each Page
def browsing_pattern():
    st.header("Browsing Pattern - Product Pair Heatmap")

    # Generate product pairs and counts
    products_per_transaction = df_item.groupby('Transaction ID')['Product ID'].apply(list)
    product_pairs = []
    for products in products_per_transaction:
        if len(products) > 1:
            product_pairs.extend(list(combinations(products, 2)))

    pair_counts = Counter(product_pairs)
    df_pair_counts = pd.DataFrame(pair_counts.items(), columns=['Product Pair', 'Count'])
    df_pair_counts[['Product 1', 'Product 2']] = pd.DataFrame(df_pair_counts['Product Pair'].tolist(), index=df_pair_counts.index)

    # Generate Heatmap
    pivot_table = df_pair_counts.pivot_table(index='Product 1', columns='Product 2', values='Count', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot_table, annot=False, cmap="coolwarm", fmt='.0f', ax=ax)
    st.pyplot(fig)

    # Display Top Cross-Selling Products
    st.header("Best to Least Cross-Selling Products")

    # Get top 10 products for each product
    top_cross_selling = {}
    for product in pivot_table.index:
        sorted_products = pivot_table.loc[product].sort_values(ascending=False).head(10)
        top_cross_selling[product] = sorted_products.index.tolist()

    # Convert dictionary to a DataFrame for display
    cross_selling_df = pd.DataFrame.from_dict(top_cross_selling, orient='index', columns=[f"Rank {i+1}" for i in range(10)])
    cross_selling_df.index.name = "Product"
    cross_selling_df.reset_index(inplace=True)

    # Display as a styled table
    st.table(cross_selling_df.style.set_caption("Top 10 Cross-Selling Products by Product"))


def seasonal_and_geographic():
    st.header("Seasonal & Geographic Analysis")

    # Merge datasets
    df_transactions_customer = pd.merge(df_transaction, df_customer, on="Customer ID", how="left")
    df_transactions_customer_item = pd.merge(df_transactions_customer, df_item, on="Transaction ID", how="left")
    df_full = pd.merge(df_transactions_customer_item, df_inventory, on='Product ID', how="left")

    # Add date-related columns
    df_full['Year of Purchase'] = pd.to_datetime(df_full['Timestamp Purchase']).dt.year
    df_full['Month of Purchase'] = pd.to_datetime(df_full['Timestamp Purchase']).dt.month
    df_full['Month Name'] = pd.to_datetime(df_full['Timestamp Purchase']).dt.strftime('%B')

    # Group data for analysis
    monthly_sales_summary = df_full.groupby(
        ['Year of Purchase', 'Month of Purchase', 'Product Category', 'Geographical Location']
    ).agg({'Units Sold': 'sum'}).reset_index()

    max_units_sold = monthly_sales_summary['Units Sold'].max()
    monthly_sales_summary['Bubble Size'] = (monthly_sales_summary['Units Sold'] / max_units_sold) * 80

    # Define regions and color map
    regions = monthly_sales_summary['Geographical Location'].unique()
    color_palette = sns.color_palette("Set2", len(regions))
    color_map = dict(zip(regions, color_palette))

    # Dropdown menu to select year
    selected_year = st.selectbox("Select Year for Analysis", options=[2022, 2023, 2024])

    # Generate the bubble chart for the selected year
    def plot_sales_for_year(year):
        year_data = monthly_sales_summary[monthly_sales_summary['Year of Purchase'] == year]
        year_data['x_positions'] = year_data['Month of Purchase']

        fig = go.Figure()
        for region in regions:
            region_data = year_data[year_data['Geographical Location'] == region]
            fig.add_trace(go.Scatter(
                x=region_data['x_positions'],
                y=region_data['Product Category'],
                mode='markers',
                marker=dict(
                    size=region_data['Bubble Size'],
                    color=f"rgb({color_map[region][0]*255}, {color_map[region][1]*255}, {color_map[region][2]*255})",
                    opacity=0.8,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name=region,
                text=[f"Region: {region}<br>Month: {month}<br>Category: {category}<br>Units Sold: {units}"
                      for month, category, units in zip(
                          region_data['Month of Purchase'],
                          region_data['Product Category'],
                          region_data['Units Sold']
                      )],
                hoverinfo="text"
            ))

        fig.update_layout(
            title=f"Monthly Sales by Product Category for Year {year}",
            xaxis=dict(
                title="Month of Purchase",
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                range=[0, 13],
            ),
            yaxis=dict(
                title="Product Category",
                categoryorder="array",
                categoryarray=monthly_sales_summary['Product Category'].unique(),
            ),
            legend_title="Region",
            width=1200,
            height=700,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        return fig

    st.subheader(f"Monthly Sales - Year {selected_year}")
    st.plotly_chart(plot_sales_for_year(selected_year), use_container_width=True)

    # "Best Selling Months and Categories" Table
    st.subheader("Best Selling Months and Categories")

    best_selling = df_full[df_full['Year of Purchase'] == selected_year].groupby(
        ['Geographical Location', 'Month Name', 'Product Category']
    )['Units Sold'].sum().reset_index()

    best_selling['Rank'] = best_selling.groupby('Geographical Location')['Units Sold'].rank(method='dense', ascending=False)

    # Filter top categories for each region
    top_categories = best_selling[best_selling['Rank'] <= 3].sort_values(by=['Geographical Location', 'Rank'])

    # Display regional tables
    for region in regions:
        st.subheader(f"Region: {region}")
        regional_data = top_categories[top_categories['Geographical Location'] == region]
        st.table(
            regional_data.pivot_table(
                index='Month Name',
                columns='Rank',
                values='Product Category',
                aggfunc=lambda x: ' & '.join(x) if isinstance(x, pd.Series) else x
            ).fillna('-')
        )


def funnel_analysis():
    st.header("Funnel Analysis")
    
    # Define funnel stages
    funnel_stages = ['Homepage', 'Product Detail Page', 'Cart Page', 'Checkout Page', 'Order Confirmation Page']
    
    # Check stages visited for each session
    def check_funnel_stages(pages_visited):
        return {stage: stage in pages_visited for stage in funnel_stages}
    
    df_behavioral['Funnel Stages'] = df_behavioral['Pages Visited'].apply(check_funnel_stages)
    funnel_stage_df = pd.DataFrame(df_behavioral['Funnel Stages'].tolist())
    funnel_stage_counts = funnel_stage_df.sum().values
    
    # Total sessions and conversion/drop-off rates
    total_sessions = len(df_behavioral)
    conversion_rates = (funnel_stage_counts / total_sessions) * 100
    dropoff_rates = [0] + [100 - (conversion_rates[i] / conversion_rates[i-1] * 100) if conversion_rates[i-1] != 0 else 0 for i in range(1, len(conversion_rates))]
    
    # Funnel graph
    index = np.arange(len(funnel_stages))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(index, conversion_rates, bar_width, label='Conversion Rate', color='skyblue')
    ax.plot(index, dropoff_rates, label='Drop-Off Rate', color='red', marker='o', linestyle='-', linewidth=2)
    
    # Annotate conversion and drop-off rates on the graph
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2, f'{conversion_rates[i]:.2f}%',
                ha='center', color='black', fontsize=10)
    for i in range(len(dropoff_rates)):
        ax.text(index[i], dropoff_rates[i] + 2, f'{dropoff_rates[i]:.2f}%',
                ha='center', color='red', fontsize=10)
    
    # Graph styling
    ax.set_xlabel('Funnel Stage')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Conversion Rate with Drop-Off Rate Overlay')
    ax.set_xticks(index)
    ax.set_xticklabels(funnel_stages, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)
    
    # Drop-Off Rate Table
    st.subheader("Drop-Off Rate (%) and Number of Customers")
    funnel_summary_df = pd.DataFrame({
        "Funnel Stage": funnel_stages,
        "Customers": funnel_stage_counts,
        "Conversion Rate (%)": conversion_rates,
        "Drop-Off Rate (%)": dropoff_rates
    })
    st.table(funnel_summary_df.style.set_caption("Drop-Off Rate and Customers at Each Funnel Stage"))


def inventory_analysis():
    st.header("Inventory Analysis")

    # Declare df_inventory as global to ensure proper access
    global df_inventory

    # Ensure df_inventory is loaded and contains required columns
    if df_inventory is None or 'Product ID' not in df_inventory.columns or 'Current Stock' not in df_inventory.columns:
        st.error("The inventory dataset is not loaded properly or missing required columns.")
        return

    # Ensure Timestamp Purchase is in datetime format
    df_transaction['Timestamp Purchase'] = pd.to_datetime(df_transaction['Timestamp Purchase'])
    df_item['Date'] = df_transaction['Timestamp Purchase'].dt.date

    # Calculate daily sales and average daily sales
    daily_sales = df_item.groupby(['Product ID', 'Date'])['Units Sold'].sum().reset_index()
    average_daily_sales = daily_sales.groupby('Product ID')['Units Sold'].mean().reset_index()
    average_daily_sales.columns = ['Product ID', 'Average Daily Sales']

    # Merge average daily sales into inventory dataframe
    df_inventory = df_inventory.merge(average_daily_sales, on='Product ID', how='left')

    # Add a column for reorder points
    df_inventory['Reorder Point'] = df_inventory['Average Daily Sales'] * 3  # Example: 3 days of average sales

    # Fill missing values for current stock and reorder point
    df_inventory['Current Stock'] = df_inventory['Current Stock'].fillna(0)
    df_inventory['Reorder Point'] = df_inventory['Reorder Point'].fillna(0)

    # SARIMAX Forecast Function
    def forecast_daily_sales(product_id, forecast_days=30):
        product_data = daily_sales[daily_sales['Product ID'] == product_id][['Date', 'Units Sold']].set_index('Date')

        if len(product_data) < 10:  # Ensure enough data
            return pd.Series([0] * forecast_days, index=pd.date_range(product_data.index.max() + timedelta(days=1), periods=forecast_days))

        model = SARIMAX(product_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))  # Weekly seasonality
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=forecast_days)
        return forecast.predicted_mean

    forecast_days = 30
    forecasted_sales = {product_id: forecast_daily_sales(product_id, forecast_days).sum() for product_id in df_inventory['Product ID']}
    df_inventory['Forecasted Sales'] = df_inventory['Product ID'].map(forecasted_sales)

    # Calculate additional units needed
    df_inventory['Additional Units Needed'] = df_inventory['Forecasted Sales'] - df_inventory['Current Stock']
    df_inventory['Additional Units Needed'] = df_inventory['Additional Units Needed'].apply(lambda x: max(x, 0))

    # Determine colors for bars
    bar_colors = ['red' if stock < reorder else 'skyblue' for stock, reorder in zip(df_inventory['Current Stock'], df_inventory['Reorder Point'])]

    # Plot Inventory Levels and Forecasted Needs
    fig, ax = plt.subplots(figsize=(18, 8))
    product_ids = df_inventory['Product ID']

    ax.bar(
        product_ids,
        df_inventory['Current Stock'],
        color=bar_colors,
        alpha=0.7,
        label='Current Stock'
    )

    ax.bar(
        product_ids,
        df_inventory['Additional Units Needed'],
        bottom=df_inventory['Current Stock'],
        color='orange',
        alpha=0.7,
        label='Additional Units Needed'
    )

    ax.plot(
        product_ids,
        df_inventory['Reorder Point'],
        color='gray',
        marker='o',
        label='Reorder Point'
    )

    for i, product_id in enumerate(product_ids):
        current_stock = df_inventory.loc[i, 'Current Stock']
        additional_units = df_inventory.loc[i, 'Additional Units Needed']
        reorder_point = df_inventory.loc[i, 'Reorder Point']

        ax.text(
            i,
            current_stock + additional_units + 1,
            f"{int(current_stock)}",
            ha='center',
            fontsize=8
        )
        ax.text(
            i,
            reorder_point + 1,
            f"{int(reorder_point)}",
            ha='center',
            color='gray',
            fontsize=8
        )

    ax.set_xticks(range(len(product_ids)))
    ax.set_xticklabels(product_ids, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Products', fontsize=10, fontweight='bold')
    ax.set_ylabel('Units', fontsize=10, fontweight='bold')
    ax.set_title('Inventory Levels and Forecasted Needs', fontsize=12, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Inventory Forecast and Analysis")
    st.table(df_inventory[['Product ID', 'Product Name', 'Current Stock', 'Reorder Point', 'Forecasted Sales', 'Additional Units Needed']])



def price_sensitivity():
    st.header("Price Sensitivity Analysis")

    # Merge datasets
    df_merged = pd.merge(df_item, df_inventory, on="Product ID", how="left")
    df_merged = pd.merge(df_merged, df_transaction, on="Transaction ID", how="left")
    df_merged = pd.merge(df_merged, df_campaign, on="Transaction ID", how="left")

    df_merged['Timestamp Purchase'] = pd.to_datetime(df_merged['Timestamp Purchase'])

    def classify_proximity_period(purchase_date, promo_dates):
        for start_date, end_date in promo_dates:
            three_months_before = start_date - timedelta(days=90)
            one_month_before = start_date - timedelta(days=30)
            one_month_after = end_date + timedelta(days=30)
            three_months_after = end_date + timedelta(days=90)

            if three_months_before <= purchase_date < one_month_before:
                return '-3M'
            elif one_month_before <= purchase_date < start_date:
                return '-1M'
            elif start_date <= purchase_date <= end_date:
                return 'During Promo'
            elif end_date < purchase_date <= one_month_after:
                return '+1M'
            elif one_month_after < purchase_date <= three_months_after:
                return '+3M'
        return 'No Promotion'

    def prepare_promotion_data(df, promo_type):
        promo_data = df[df['Promotion Type'] == promo_type]
        promo_dates = promo_data['Promotional Period'].dropna().unique()
        dates = []
        for period in promo_dates:
            start_date, end_date = [datetime.strptime(date.strip(), "%Y-%m-%d") for date in period.split('to')]
            dates.append((start_date, end_date))
        df['Promotion Period Classification'] = df['Timestamp Purchase'].apply(lambda x: classify_proximity_period(x, dates))
        return df.groupby(['Promotion Period Classification', 'Product Category'])['Units Sold'].sum().reset_index()

    def plot_promotion_data(aggregated_df, promo_type):
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=aggregated_df,
            x='Promotion Period Classification',
            y='Units Sold',
            hue='Product Category',
            palette='Set2',
            errorbar=None
        )
        plt.title(f"Units Sold by Promotion Period - {promo_type}")
        plt.xlabel("Promotion Period")
        plt.ylabel("Units Sold")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(plt)

    # Generate tables showing best-to-worst promotions for each category
    def generate_category_promo_tables(df):
        categories = df['Product Category'].unique()
        for category in categories:
            category_data = df[df['Product Category'] == category]
            promo_ranking = category_data.groupby(['Promotion Type'])['Units Sold'].sum().reset_index()
            promo_ranking = promo_ranking.sort_values(by='Units Sold', ascending=False)

            st.subheader(f"Promotions Ranked by Effectiveness - {category}")
            st.table(
                promo_ranking.rename(columns={'Units Sold': 'Total Units Sold'})
                .style.format({'Total Units Sold': '{:.0f}'})
            )

    promotion_types = df_merged['Promotion Type'].unique()

    # Iterate through each promotion type, plot data, and display tables
    for promo_type in promotion_types:
        aggregated = prepare_promotion_data(df_merged, promo_type)
        st.subheader(f"Promotion Type: {promo_type}")
        plot_promotion_data(aggregated, promo_type)

    # Generate and display category-specific tables
    generate_category_promo_tables(df_merged)



# Conditional Navigation Logic
if page == "Browsing Pattern":
    browsing_pattern()
elif page == "Seasonal & Geographic":
    seasonal_and_geographic()
elif page == "Price Sensitivity":
    price_sensitivity()
elif page == "Funnel Analysis":
    funnel_analysis()
elif page == "Inventory Analysis":
    inventory_analysis()
