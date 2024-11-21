import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go

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
    products_per_transaction = df_item.groupby('Transaction ID')['Product ID'].apply(list)
    product_pairs = []
    for products in products_per_transaction:
        if len(products) > 1:
            product_pairs.extend(list(combinations(products, 2)))

    pair_counts = Counter(product_pairs)
    df_pair_counts = pd.DataFrame(pair_counts.items(), columns=['Product Pair', 'Count'])
    df_pair_counts[['Product 1', 'Product 2']] = pd.DataFrame(df_pair_counts['Product Pair'].tolist(), index=df_pair_counts.index)

    pivot_table = df_pair_counts.pivot_table(index='Product 1', columns='Product 2', values='Count', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot_table, annot=False, cmap="coolwarm", fmt='.0f', ax=ax)
    st.pyplot(fig)

def seasonal_and_geographic():
    st.header("Seasonal & Geographic Analysis")
    df_transactions_customer = pd.merge(df_transaction, df_customer, on="Customer ID", how="left")
    df_transactions_customer_item = pd.merge(df_transactions_customer, df_item, on="Transaction ID", how="left")
    df_full = pd.merge(df_transactions_customer_item, df_inventory, on='Product ID', how="left")

    df_full['Year of Purchase'] = pd.to_datetime(df_full['Timestamp Purchase']).dt.year
    df_full['Month of Purchase'] = pd.to_datetime(df_full['Timestamp Purchase']).dt.month
    df_full['Day Fraction'] = pd.to_datetime(df_full['Timestamp Purchase']).dt.day / pd.to_datetime(df_full['Timestamp Purchase']).dt.days_in_month

    monthly_sales_summary = df_full.groupby(
        ['Year of Purchase', 'Month of Purchase', 'Product Category', 'Geographical Location']
    ).agg({'Units Sold': 'sum', 'Day Fraction': 'mean'}).reset_index()

    max_units_sold = monthly_sales_summary['Units Sold'].max()
    monthly_sales_summary['Bubble Size'] = (monthly_sales_summary['Units Sold'] / max_units_sold) * 80

    regions = monthly_sales_summary['Geographical Location'].unique()
    color_palette = sns.color_palette("Set2", len(regions))
    color_map = dict(zip(regions, color_palette))

    def plot_sales_for_year(year):
        year_data = monthly_sales_summary[monthly_sales_summary['Year of Purchase'] == year]
        year_data['x_positions'] = year_data['Month of Purchase'] + year_data['Day Fraction'] - 1

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

    st.subheader("Monthly Sales - Year 2022")
    st.plotly_chart(plot_sales_for_year(2022), use_container_width=True)

    st.subheader("Monthly Sales - Year 2023")
    st.plotly_chart(plot_sales_for_year(2023), use_container_width=True)

    st.subheader("Monthly Sales - Year 2024")
    st.plotly_chart(plot_sales_for_year(2024), use_container_width=True)

def price_sensitivity():
    st.header("Price Sensitivity Analysis")
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

    promotion_types = df_merged['Promotion Type'].unique()
    for promo_type in promotion_types:
        aggregated = prepare_promotion_data(df_merged, promo_type)
        st.subheader(f"Promotion Type: {promo_type}")
        plot_promotion_data(aggregated, promo_type)

def funnel_analysis():
    st.header("Funnel Analysis")
    funnel_stages = ['Homepage', 'Product Detail Page', 'Cart Page', 'Checkout Page', 'Order Confirmation Page']

    def check_funnel_stages(pages_visited):
        return {stage: stage in pages_visited for stage in funnel_stages}

    df_behavioral['Funnel Stages'] = df_behavioral['Pages Visited'].apply(check_funnel_stages)
    funnel_stage_df = pd.DataFrame(df_behavioral['Funnel Stages'].tolist())
    funnel_stage_counts = funnel_stage_df.sum()

    total_sessions = len(df_behavioral)
    conversion_rates = (funnel_stage_counts / total_sessions) * 100
    dropoff_rates = 100 - conversion_rates

    stages = funnel_stage_counts.index
    bar_width = 0.35
    index = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(index, conversion_rates, bar_width, label='Conversion Rate', color='skyblue')
    ax.plot(index, dropoff_rates, label='Drop-Off Rate', color='red', marker='o', linestyle='-', linewidth=2)

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2, f'{conversion_rates.iloc[i]:.2f}%',
                ha='center', color='black', fontsize=10)
    for i in range(len(dropoff_rates)):
        ax.text(index[i], dropoff_rates.iloc[i] + 2, f'{dropoff_rates.iloc[i]:.2f}%',
                ha='center', color='red', fontsize=10)

    ax.set_xlabel('Funnel Stage')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Conversion Rate with Drop-Off Rate Overlay')
    ax.set_xticks(index)
    ax.set_xticklabels(stages, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

def inventory_analysis():
    st.header("Inventory Analysis")
    global df_inventory

    df_transaction['Timestamp Purchase'] = pd.to_datetime(df_transaction['Timestamp Purchase'])
    df_item['Date'] = df_transaction['Timestamp Purchase'].dt.date

    daily_sales = df_item.groupby(['Product ID', 'Date'])['Units Sold'].sum().reset_index()
    average_daily_sales = daily_sales.groupby('Product ID')['Units Sold'].mean().reset_index()
    average_daily_sales.columns = ['Product ID', 'Average Daily Sales']

    df_inventory = df_inventory.merge(average_daily_sales, on='Product ID', how='left')

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.bar(df_inventory['Product ID'], df_inventory['Current Stock'], color='skyblue', alpha=0.7)
    ax.axhline(df_inventory['Current Stock'].mean(), color='red', linestyle='--', label='Reorder Point')
    ax.set_title("Inventory Levels and Reorder Points")
    ax.set_xlabel("Products")
    ax.set_ylabel("Stock Level")
    st.pyplot(fig)

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
