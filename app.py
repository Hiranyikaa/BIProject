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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Hardcoded usernames and passwords (for demo purposes)
USER_CREDENTIALS = {
    "admin": "admin123",
    "user1": "password1",
    "user2": "password2",
}

# Add Custom CSS for Frontend
st.markdown("""
    <style>
        body { font-family: 'Poppins', sans-serif; background-color: #f8f8f8; }
        h1, h2, h3 { color: #5a5a5a; }
        .logout-button { background-color: #F899CE; color: white; border: none; border-radius: 5px; padding: 6px 12px; }
        .sidebar .sidebar-content { background: linear-gradient(120deg, #A49CF4, #F899CE); color: white; }
    </style>

    <style>
        body { 
            font-family: 'Poppins', sans-serif; 
            background-color: #F8F9FA; 
            color: #6C3483; 
        }
        h1, h2, h3 { 
            color: #6C3483; 
            font-weight: bold; 
        }
        .sidebar .sidebar-content { 
            background: linear-gradient(120deg, #A49CF4, #F899CE); 
            color: white; 
        }
        .stButton>button { 
            background-color: #C39BD3; 
            color: white; 
            border-radius: 5px; 
            border: none; 
            font-weight: bold; 
            padding: 8px 12px; 
        }
        .stDataFrame { 
            background-color: #EBDEF0; 
            color: #6C3483; 
            font-weight: bold; 
        }
        .stMarkdown, .stTextInput>div>div>input { 
            color: #6C3483; 
        }
        .stNumberInput>div>input { 
            background-color: #F9EBEA; 
            color: #6C3483; 
            border: 1px solid #D2B4DE; 
            border-radius: 5px; 
            padding: 5px; 
        }
    </style>
""", unsafe_allow_html=True)

# Load datasets
df_behavioral = pd.read_csv(r'Behavioral_Dataset.csv')
df_customer = pd.read_csv(r'Customer_Dataset.csv')
df_item = pd.read_csv(r'Transaction_Items_Dataset.csv')
df_transaction = pd.read_csv(r'Transaction_Dataset.csv')
df_inventory = pd.read_csv(r'Inventory_Dataset.csv')
df_campaign = pd.read_csv(r'Campaign_Dataset.csv')

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None


# Function: Login System
def login():
    """Login page function."""
    if not st.session_state["authenticated"]:
        st.sidebar.title("Login")  # แสดง "Login" ใน Sidebar เฉพาะเมื่อยังไม่ได้ล็อกอิน

        # Login form
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button:
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success(f"Logged in as: {username}")
            else:
                st.error("Invalid username or password.")
        return False

    return True


def logout():
    """Logout function."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None


# Function Definitions for Each Page
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter
import streamlit as st

def browsing_pattern():
    st.header("Browsing Pattern - Product Pair Heatmap")

    # Generate product pairs and counts
    products_per_transaction = df_item.groupby('Transaction ID')['Product ID'].apply(list)

    product_pairs = []
    for products in products_per_transaction:
        if len(products) > 1:
            sorted_products = sorted(products)
            product_pairs.extend(list(combinations(sorted_products, 2)))

    pair_counts = Counter(product_pairs)

    df_pair_counts = pd.DataFrame(pair_counts.items(), columns=['Product Pair', 'Count'])
    df_pair_counts[['Product 1', 'Product 2']] = pd.DataFrame(df_pair_counts['Product Pair'].tolist(), index=df_pair_counts.index)

    # Generate pivot table
    pivot_table = df_pair_counts.pivot_table(index='Product 1', columns='Product 2', values='Count', fill_value=0)
    pivot_table = pivot_table.round(0).astype(int)

    # Set lower triangular values to 0
    for i in range(len(pivot_table)):
        for j in range(i):
            pivot_table.iloc[i, j] = 0

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt='d', ax=ax, cbar_kws={'label': 'Number of Pairs'})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Product Pair Heatmap: Frequently Bought Together')
    plt.tight_layout()
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

    # Add Distribution Plot for Product Pair Counts
    st.header("Distribution of Product Pair Counts")
    Q1 = df_pair_counts['Count'].quantile(0.25)
    Q3 = df_pair_counts['Count'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    plt.figure(figsize=(10, 6))
    sns.histplot(df_pair_counts['Count'], bins=30, kde=True, color="skyblue", alpha=0.8)
    plt.axvline(upper_bound, color='r', linestyle='--', label='Outstanding Threshold')
    plt.title('Distribution of Product Pair Counts with Outstanding Threshold', fontsize=14)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    st.pyplot(plt)

    # Display All Product Pairs in a Table
    st.header("All Product Pairs and Counts")
    st.dataframe(df_pair_counts[['Product Pair', 'Count', 'Product 1', 'Product 2']])


def forecast_sales_by_region_and_category(df, forecast_months=3):
    """
    Forecast sales by region and product category using all available data across years.
    Dynamically starts forecasting from the latest available date in the dataset.
    """
    forecast_results = []

    # Determine the latest year and month in the dataset
    latest_year = df['Year of Purchase'].max()
    latest_month = df[df['Year of Purchase'] == latest_year]['Month of Purchase'].max()

    # Group data by region and product category
    grouped = df.groupby(['Geographical Location', 'Product Category'])

    for (region, category), data in grouped:
        # Aggregate monthly sales across all years
        monthly_sales = data.groupby(['Year of Purchase', 'Month of Purchase'])['Units Sold'].sum()
        monthly_sales = monthly_sales.unstack(level=0).stack().reindex(
            pd.MultiIndex.from_product(
                [range(1, 13), sorted(df['Year of Purchase'].unique())],
                names=['Month', 'Year']
            ),
            fill_value=0
        )

        # Flatten the index for ExponentialSmoothing
        monthly_sales = monthly_sales.reset_index(level=1, drop=True)
        try:
            # Create a proper time index for the monthly sales
            start_date = f"{df['Year of Purchase'].min()}-01-01"
            monthly_sales.index = pd.date_range(
                start=start_date,
                periods=len(monthly_sales),
                freq='M'  # Month-end frequency
            )

            # Fit Holt-Winters model using all historical data
            model = ExponentialSmoothing(monthly_sales, seasonal='add', seasonal_periods=12)
            fit = model.fit()
        except Exception as e:
            logging.warning(f"Insufficient data or error for region: {region}, category: {category}. Using fallback model. Error: {e}")
            # Fallback to a simple average
            forecast = [monthly_sales.mean()] * forecast_months
            forecast_df = pd.DataFrame({
                'Geographical Location': region,
                'Product Category': category,
                'Month of Purchase': range(latest_month + 1, latest_month + 1 + forecast_months),
                'Units Sold': forecast
            })
            forecast_results.append(forecast_df)
            continue

        # Forecast next months dynamically
        forecast = fit.forecast(forecast_months)

        # Handle year-end wrap-around for forecasted months
        forecast_months_adjusted = []
        current_month = latest_month
        current_year = latest_year
        for _ in range(forecast_months):
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
            forecast_months_adjusted.append((current_year, current_month))

        forecast_df = pd.DataFrame({
            'Geographical Location': region,
            'Product Category': category,
            'Year of Purchase': [x[0] for x in forecast_months_adjusted],
            'Month of Purchase': [x[1] for x in forecast_months_adjusted],
            'Units Sold': forecast
        })
        forecast_results.append(forecast_df)

    # Combine all forecasts
    if forecast_results:
        forecast_data = pd.concat(forecast_results, ignore_index=True)
    else:
        forecast_data = pd.DataFrame(columns=['Geographical Location', 'Product Category', 'Year of Purchase', 'Month of Purchase', 'Units Sold'])

    return forecast_data


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

    # Get the latest year from the data
    latest_year = df_full['Year of Purchase'].max()

    # Dropdown menu to select year with default as the latest year
    selected_year = st.selectbox(
        "Select Year for Analysis",
        options=sorted(df_full['Year of Purchase'].unique()),
        index=list(sorted(df_full['Year of Purchase'].unique())).index(latest_year)
    )

    # Add input for Percentile threshold
    st.sidebar.subheader("Percentile Filter")
    percentile_threshold = st.sidebar.slider("Select Percentile (%)", min_value=0, max_value=100, value=90, step=1)

    # Filter data based on Percentile
    threshold_value = np.percentile(monthly_sales_summary['Units Sold'], percentile_threshold)
    filtered_sales_summary = monthly_sales_summary[monthly_sales_summary['Units Sold'] >= threshold_value]

    # Add option to include forecast
    include_forecast = st.sidebar.checkbox("Include Forecast Data", value=False)

    # Generate forecast if selected
    forecast_data = None
    if include_forecast:
        forecast_months = st.sidebar.slider("Forecast Months", min_value=1, max_value=12, value=3, step=1)
        forecast_data = forecast_sales_by_region_and_category(df_full, forecast_months)

    # Generate the bubble chart for the selected year
    def plot_sales_with_forecast(year, forecast_df=None):
        year_data = filtered_sales_summary[filtered_sales_summary['Year of Purchase'] == year]
        year_data['x_positions'] = year_data['Month of Purchase']

        fig = go.Figure()

        # Add historical data to the plot
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
                name=f"{region} (Historical)",
                text=[f"Region: {region}<br>Month: {month}<br>Category: {category}<br>Units Sold: {units}"
                      for month, category, units in zip(
                          region_data['Month of Purchase'],
                          region_data['Product Category'],
                          region_data['Units Sold']
                      )],
                hoverinfo="text"
            ))

        # Add forecast data to the plot
        if forecast_df is not None:
            forecast_year_data = forecast_df[forecast_df['Year of Purchase'] == year]
            for region in regions:
                forecast_region_data = forecast_year_data[forecast_year_data['Geographical Location'] == region]
                fig.add_trace(go.Scatter(
                    x=forecast_region_data['Month of Purchase'],
                    y=forecast_region_data['Product Category'],
                    mode='markers',
                    marker=dict(
                        size=forecast_region_data['Units Sold'] / max_units_sold * 80,
                        color=f"rgb({color_map[region][0]*255}, {color_map[region][1]*255}, {color_map[region][2]*255})",
                        opacity=0.4,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name=f"{region} (Forecast)",
                    text=[f"Region: {region}<br>Month: {month}<br>Category: {category}<br>Units Sold: {units}"
                          for month, category, units in zip(
                              forecast_region_data['Month of Purchase'],
                              forecast_region_data['Product Category'],
                              forecast_region_data['Units Sold']
                          )],
                    hoverinfo="text",
                    showlegend=False  # Avoid duplicate legend entries
                ))

        fig.update_layout(
            title=f"Monthly Sales with Forecast (Year: {year})",
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

    st.plotly_chart(plot_sales_with_forecast(selected_year, forecast_data), use_container_width=True)

    # **New Table**: Average Units Sold Summary
    st.subheader("Average Units Sold by Year, Region, and Product Category")
    avg_units_sold_summary = df_full.groupby(
        ['Year of Purchase', 'Geographical Location', 'Product Category']
    ).agg({'Units Sold': 'mean'}).reset_index()

    avg_units_sold_summary.rename(columns={'Units Sold': 'Avg Units Sold'}, inplace=True)
    avg_units_sold_summary = avg_units_sold_summary.sort_values(
        ['Year of Purchase', 'Geographical Location', 'Avg Units Sold'], ascending=[True, True, False]
    )

    # **Checkbox Filters**
    st.sidebar.subheader("Filters")
    selected_years = st.sidebar.multiselect(
        "Select Year(s)", 
        options=avg_units_sold_summary['Year of Purchase'].unique(), 
        default=avg_units_sold_summary['Year of Purchase'].unique()
    )
    selected_regions = st.sidebar.multiselect(
        "Select Region(s)", 
        options=avg_units_sold_summary['Geographical Location'].unique(), 
        default=avg_units_sold_summary['Geographical Location'].unique()
    )
    selected_categories = st.sidebar.multiselect(
        "Select Product Category(ies)", 
        options=avg_units_sold_summary['Product Category'].unique(), 
        default=avg_units_sold_summary['Product Category'].unique()
    )

    # Apply filters
    filtered_summary = avg_units_sold_summary[
        (avg_units_sold_summary['Year of Purchase'].isin(selected_years)) &
        (avg_units_sold_summary['Geographical Location'].isin(selected_regions)) &
        (avg_units_sold_summary['Product Category'].isin(selected_categories))
    ]

    # Display the filtered table
    st.table(filtered_summary)
    

def funnel_analysis():
    st.header("Funnel Analysis")

    # Define funnel stages
    funnel_stages = ['Homepage', 'Product Detail Page', 'Cart Page', 'Checkout Page', 'Order Confirmation Page']

    # Check which stages each session reached
    def check_funnel_stages(pages_visited):
        return {stage: stage in pages_visited for stage in funnel_stages}

    # Apply the function to the dataset
    df_behavioral['Funnel Stages'] = df_behavioral['Pages Visited'].apply(check_funnel_stages)

    # Create a DataFrame for funnel stages
    funnel_stage_df = pd.DataFrame(df_behavioral['Funnel Stages'].tolist())

    # Count how many sessions reached each stage
    funnel_stage_counts = funnel_stage_df.sum()

    # Calculate conversion rates
    total_sessions = len(df_behavioral)
    conversion_rates = (funnel_stage_counts / total_sessions) * 100

    # Calculate drop-off rates
    dropoff_rates = 100 - conversion_rates

    # Plot the funnel analysis
    index = np.arange(len(funnel_stages))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot conversion rates as bars
    bars = ax.bar(index, conversion_rates, bar_width, label='Conversion Rate', color='skyblue')

    # Plot drop-off rates as a line
    ax.plot(index, dropoff_rates, label='Drop-Off Rate', color='red', marker='o', linestyle='-', linewidth=2)

    # Annotate conversion rates on bars
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
            f'{conversion_rates[i]:.2f}%',
            ha='center', color='black', fontsize=10
        )

    # Annotate drop-off rates on line
    for i, rate in enumerate(dropoff_rates):
        ax.text(
            index[i], rate + 2,
            f'{rate:.2f}%',
            ha='center', color='red', fontsize=10
        )

    # Style the plot
    ax.set_xlabel('Funnel Stage', fontweight='bold')
    ax.set_ylabel('Rate (%)', fontweight='bold')
    ax.set_title('Conversion Rate with Drop-Off Rate Overlay', fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(funnel_stages, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    st.pyplot(fig)

    # Display Drop-Off Rate Table
    st.subheader("Drop-Off Rate (%) and Number of Customers")
    funnel_summary_df = pd.DataFrame({
        "Funnel Stage": funnel_stages,
        "Customers": funnel_stage_counts.values,
        "Conversion Rate (%)": conversion_rates.values,
        "Drop-Off Rate (%)": dropoff_rates.values
    })
    st.table(funnel_summary_df.style.set_caption("Drop-Off Rate and Customers at Each Funnel Stage"))

    # Add User Segmentation Analysis
    st.subheader("User Segmentation by Funnel Progress")

    def segment_sessions(funnel_stages):
        if funnel_stages['Order Confirmation Page']:
            return 'Converted Users'
        elif funnel_stages['Cart Page']:
            return 'Engaged Users'
        else:
            return 'Abandoned Users'

    df_behavioral['Segment'] = df_behavioral['Funnel Stages'].apply(segment_sessions)
    segment_counts = df_behavioral['Segment'].value_counts()

    # Create a Pie Chart for Segments
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    colors = ['skyblue', 'orange', 'lightgreen']
    explode = (0.1, 0, 0)  # Emphasize Converted Users
    ax_pie.pie(
        segment_counts,
        labels=segment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        explode=explode,
        colors=colors
    )
    ax_pie.set_title('Session Segmentation by Funnel Progress', fontsize=14, fontweight='bold')
    st.pyplot(fig_pie)


def inventory_analysis(): 
    st.header("Inventory Analysis")

    global df_inventory, df_item, df_transaction

    # Ensure datasets are properly loaded
    if df_inventory is None or df_item is None or df_transaction is None:
        st.error("Datasets are not loaded properly.")
        return

    # Merge transaction data with item data
    df_transaction['Timestamp Purchase'] = pd.to_datetime(df_transaction['Timestamp Purchase'])
    df_item = df_item.merge(df_transaction[['Transaction ID', 'Timestamp Purchase']], on='Transaction ID', suffixes=('', '_Transaction'))
    df_item['Date'] = df_item['Timestamp Purchase'].dt.date

    # Calculate daily sales and average daily sales
    daily_sales = df_item.groupby(['Product ID', 'Date'])['Units Sold'].sum().reset_index()
    average_daily_sales = daily_sales.groupby('Product ID')['Units Sold'].mean().reset_index()
    average_daily_sales.columns = ['Product ID', 'Average Daily Sales']

    # Merge average daily sales into inventory dataframe
    df_inventory = df_inventory.merge(average_daily_sales, on='Product ID', how='left')

    # Define lead times based on product categories
    category_lead_times = {
        'Clothing': 5,
        'Sports & Outdoors': 7,
        'Home & Kitchen': 10,
        'Electronics': 15,
        'Personal Care': 3,
        'Kid Products': 6,
        'Grocery': 2,
        'Books': 8,
        'Pet Supplies': 5
    }
    df_inventory['Lead Time'] = df_inventory['Product Category'].map(category_lead_times)

    # Calculate Reorder Point
    df_inventory['Reorder Point'] = df_inventory['Average Daily Sales'] * df_inventory['Lead Time']

    # Stock status calculation
    def determine_stock_status(row):
        if row['Current Stock'] >= row['Reorder Point']:
            return 'In Stock'
        elif 0 < row['Current Stock'] < row['Reorder Point']:
            return 'Low Stock'
        else:
            return 'Out of Stock'

    df_inventory['Stock Status'] = df_inventory.apply(determine_stock_status, axis=1)

    # Sidebar to toggle between graph types
    st.sidebar.subheader("Graph Type")
    graph_type = st.sidebar.radio("Select Graph Type", options=["Historical", "Predictive"])

    # Historical and Predictive graphs
    fig, ax = plt.subplots(figsize=(20, 7))

    if graph_type == "Historical":
        bars = ax.bar(
            df_inventory['Product ID'],
            df_inventory['Current Stock'],
            color=['skyblue' if status == 'In Stock' else 'red' for status in df_inventory['Stock Status']],
            alpha=0.7,
            label='Current Stock'
        )
        ax.plot(
            df_inventory['Product ID'],
            df_inventory['Reorder Point'],
            color='gray',
            marker='o',
            linewidth=1.5,
            label='Reorder Point'
        )
    elif graph_type == "Predictive":
        # Add forecasted sales for predictive analysis
        forecast_days = 30
        forecasted_sales = {product_id: forecast_daily_sales(product_id, daily_sales, forecast_days)['Forecasted Sales'].sum()
                            for product_id in df_inventory['Product ID']}
        df_inventory['Forecasted Sales'] = df_inventory['Product ID'].map(forecasted_sales)

        bars = ax.bar(
            df_inventory['Product ID'],
            df_inventory['Current Stock'],
            color=['skyblue' if status == 'In Stock' else 'red' for status in df_inventory['Stock Status']],
            alpha=0.7,
            label='Current Stock'
        )
        ax.bar(
            df_inventory['Product ID'],
            df_inventory['Forecasted Sales'],
            bottom=df_inventory['Current Stock'],
            color='orange',
            alpha=0.7,
            label='Forecasted Sales'
        )
        ax.plot(
            df_inventory['Product ID'],
            df_inventory['Reorder Point'],
            color='gray',
            marker='o',
            linewidth=1.5,
            label='Reorder Point'
        )

    ax.set_xlabel('Product ID', fontweight='bold', fontsize=12)
    ax.set_ylabel('Units', fontweight='bold', fontsize=12)
    ax.set_title('Inventory Levels and Reorder Points', fontweight='bold', fontsize=14)
    ax.set_xticks(range(len(df_inventory['Product ID'])))
    ax.set_xticklabels(df_inventory['Product ID'], rotation=45, ha='right', fontsize=9)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Stock Status Summary
    st.subheader("Stock Status Summary")
    stock_status_summary = df_inventory.groupby('Stock Status').agg(Number_of_Types=('Product ID', 'nunique')).reset_index()
    st.table(stock_status_summary)

    # Product Category Summary
    st.subheader("Product Category Summary")
    category_summary = df_inventory.groupby('Product Category').agg(
        Number_of_Product=('Product ID', 'nunique'),
        Average_Stock_Level=('Current Stock', 'mean')
    ).reset_index()
    st.table(category_summary)


def forecast_daily_sales(product_id, daily_sales, forecast_days=30):
    """Generate daily sales forecast for a specific product."""
    product_data = daily_sales[daily_sales['Product ID'] == product_id].set_index('Date')['Units Sold']
    if len(product_data) < 10:
        return pd.DataFrame({'Date': [], 'Forecasted Sales': []})

    model = SARIMAX(product_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=forecast_days)
    forecast_values = forecast.predicted_mean

    forecast_dates = pd.date_range(start=product_data.index.max() + timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({'Date': forecast_dates, 'Forecasted Sales': forecast_values})


def calculate_avg_units(promotion_table):
    # คำนวณค่าเฉลี่ยก่อน, ระหว่าง, และหลังโปรโมชัน
    promotion_table['Avg Before Promo'] = promotion_table[['-3M', '-1M']].mean(axis=1)
    promotion_table['Avg During Promo'] = promotion_table['During Promo']
    promotion_table['Avg After Promo'] = promotion_table[['+1M', '+3M']].mean(axis=1)
    return promotion_table[['Avg Before Promo', 'Avg During Promo', 'Avg After Promo']]

def price_sensitivity():
    st.header("Price Sensitivity Analysis")

    # Merge datasets
    df_merged = pd.merge(df_item, df_inventory, on="Product ID", how="left")
    df_merged = pd.merge(df_merged, df_transaction, on="Transaction ID", how="left")
    df_merged = pd.merge(df_merged, df_campaign, on="Transaction ID", how="left")

    df_merged['Timestamp Purchase'] = pd.to_datetime(df_merged['Timestamp Purchase'])

    # Helper function to parse promotional periods
    def get_promo_dates(df):
        promo_dates = []
        for period in df['Promotional Period'].dropna().unique():
            try:
                start_date, end_date = [datetime.strptime(date.strip(), "%Y-%m-%d") for date in period.split('to')]
                promo_dates.append((start_date, end_date))
            except ValueError:
                st.warning(f"Invalid period format: {period}")
        return promo_dates

    # Classification function
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

    # Prepare data for each promotion type
    def prepare_promotion_data(df, promo_dates):
        df['Promotion Period Classification'] = df['Timestamp Purchase'].apply(lambda x: classify_proximity_period(x, promo_dates))
        return df.groupby(['Promotion Period Classification', 'Product Category'])['Units Sold'].sum().reset_index()

    def create_promotion_table(aggregated_df):
        pivot_table = aggregated_df.pivot(
            index='Product Category',
            columns='Promotion Period Classification',
            values='Units Sold'
        ).fillna(0)  # Fill missing values with 0
        return pivot_table

    # Separate datasets by promotion type
    buy1_get1_df = df_merged[df_merged['Promotion Type'] == 'Buy 1 Get 1 Free'].copy()
    free_shipping_df = df_merged[df_merged['Promotion Type'] == 'Free Shipping'].copy()
    discount_10_df = df_merged[df_merged['Promotion Type'] == '10% Off'].copy()
    no_promotion_df = df_merged[df_merged['Promotion Type'] == 'No Promotion'].copy()

    # Get promotional periods
    buy1_get1_promo_dates = get_promo_dates(buy1_get1_df)
    free_shipping_promo_dates = get_promo_dates(free_shipping_df)
    discount_10_promo_dates = get_promo_dates(discount_10_df)

    # Aggregate data
    buy1_get1_aggregated = prepare_promotion_data(pd.concat([buy1_get1_df, no_promotion_df]), buy1_get1_promo_dates)
    free_shipping_aggregated = prepare_promotion_data(pd.concat([free_shipping_df, no_promotion_df]), free_shipping_promo_dates)
    discount_10_aggregated = prepare_promotion_data(pd.concat([discount_10_df, no_promotion_df]), discount_10_promo_dates)

    # Plotting function
    def plot_promotion_data(aggregated_df, promotion_type):
        period_order = ['-3M', '-1M', 'During Promo', '+1M', '+3M']
        aggregated_df['Promotion Period Classification'] = pd.Categorical(
            aggregated_df['Promotion Period Classification'], categories=period_order, ordered=True
        )

        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=aggregated_df,
            x='Promotion Period Classification',
            y='Units Sold',
            hue='Product Category',
            palette='Set1',
            errorbar=None
        )
        plt.title(f"Units Sold by Promotion Period and Product Category - {promotion_type}")
        plt.xlabel("Promotion Period")
        plt.ylabel("Units Sold")
        plt.legend(title="Product Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

    # Display results for each promotion type
    avg_comparison_list = []
    for promo_type, aggregated_df, promo_title in [
        ("Buy 1 Get 1 Free", buy1_get1_aggregated, "Buy 1 Get 1 Free"),
        ("Free Shipping", free_shipping_aggregated, "Free Shipping"),
        ("10% Off", discount_10_aggregated, "10% Off"),
    ]:
        st.subheader(f"Promotion Type: {promo_title}")
        plot_promotion_data(aggregated_df, promo_type)

        # Create and display the table
        promotion_table = create_promotion_table(aggregated_df)
        st.table(promotion_table)

        # Calculate and store average units
        avg_comparison_list.append(calculate_avg_units(promotion_table).mean())

    # Combine and plot comparison of promotion effectiveness
    avg_comparison = pd.concat(avg_comparison_list, axis=1)
    avg_comparison.columns = ['Buy 1 Get 1 Free', 'Free Shipping', '10% Off']

    st.subheader("Comparison of Promotion Effectiveness")
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_comparison.T.plot(kind='bar', ax=ax, rot=0, cmap='Set2')

    ax.set_title("Comparison of Promotion Effectiveness by Average Units Sold")
    ax.set_xlabel("Promotion Type")
    ax.set_ylabel("Average Units Sold")
    ax.legend(title="Promotion Period", loc="upper left", bbox_to_anchor=(1.05, 1))
    st.pyplot(fig)

    st.table(avg_comparison)


# Main App with Login Check
if login():  # User authenticated
    # Sidebar Navigation
    st.sidebar.title("Dynamic BI for Retail")
    page = st.sidebar.radio(
        "",
        options=["Browsing Pattern", "Seasonal & Geographic", "Price Sensitivity", "Funnel Analysis", "Inventory Analysis"]
    )

    # Add Logout Button to Sidebar at the Bottom
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)  # Optional Divider
    if st.sidebar.button("Logout", key="logout"):
        logout()


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
else:
    st.empty()  # ใช้แทนการแสดงข้อความใดๆ

