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
from matplotlib.colors import LinearSegmentedColormap

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


# Custom CSS to adjust header size
st.markdown("""
    <style>
            
         h2 {
            font-size: 1.5rem !important; /* ปรับขนาดของ h1 */
        }

        h3 {
            font-size: 0.7rem !important; /* Adjust subheader size */
            
        }
            
    </style>
""", unsafe_allow_html=True)


def browsing_pattern():
    st.header("Browsing Pattern Analysis")  # Main header stays as is

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

    # Custom colormap
    custom_palette = [
        '#F5EEF8', '#E8DAEF', '#D2B4DE', '#BB8FCE', '#A569BD',
        '#884EA0', '#76448A', '#633974', '#512E5F', 'red'
    ]
    cmap = LinearSegmentedColormap.from_list("CustomPurplePinkRed", custom_palette)

    # Heatmap Section
    st.markdown("### Product Pair Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap=cmap, fmt='d', ax=ax, cbar_kws={'label': 'Number of Pairs'})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    # Distribution of Product Pair Counts Section (moved below the heatmap)
    st.markdown("### Distribution of Product Pair Counts")
    Q1 = df_pair_counts['Count'].quantile(0.25)
    Q3 = df_pair_counts['Count'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    plt.figure(figsize=(8, 6))
    sns.histplot(df_pair_counts['Count'], bins=30, kde=True, color="#A569BD", alpha=0.8)
    plt.axvline(upper_bound, color='red', linestyle='--', label='Outstanding Threshold')
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    st.pyplot(plt)

    # Create another two-column layout for tables
    col3, col4 = st.columns(2)

    # Left column: Best to Least Cross-Selling Products
    with col3:
        st.markdown("### Best to Least Cross-Selling Products")
        top_cross_selling = {}
        for product in pivot_table.index:
            sorted_products = pivot_table.loc[product].sort_values(ascending=False).head(10)
            top_cross_selling[product] = sorted_products.index.tolist()

        cross_selling_df = pd.DataFrame.from_dict(top_cross_selling, orient='index', columns=[f"Rank {i+1}" for i in range(10)])
        cross_selling_df.index.name = "Product"
        cross_selling_df.reset_index(inplace=True)
        st.dataframe(cross_selling_df, height=400)

    # Right column: All Product Pairs and Counts
    with col4:
        st.markdown("### All Product Pairs and Counts")
        st.dataframe(df_pair_counts[['Product Pair', 'Count', 'Product 1', 'Product 2']])


def forecast_sales_by_region_and_category(df, forecast_months=3):
    """
    Forecast sales by region and product category using all available data across years.
    """
    forecast_results = []
    latest_year = df['Year of Purchase'].max()
    latest_month = df[df['Year of Purchase'] == latest_year]['Month of Purchase'].max()
    grouped = df.groupby(['Geographical Location', 'Product Category'])

    for (region, category), data in grouped:
        monthly_sales = data.groupby(['Year of Purchase', 'Month of Purchase'])['Units Sold'].sum()
        monthly_sales = monthly_sales.unstack(level=0).stack().reindex(
            pd.MultiIndex.from_product(
                [range(1, 13), sorted(df['Year of Purchase'].unique())],
                names=['Month', 'Year']
            ),
            fill_value=0
        )

        monthly_sales = monthly_sales.reset_index(level=1, drop=True)
        try:
            start_date = f"{df['Year of Purchase'].min()}-01-01"
            monthly_sales.index = pd.date_range(start=start_date, periods=len(monthly_sales), freq='M')
            model = ExponentialSmoothing(monthly_sales, seasonal='add', seasonal_periods=12)
            fit = model.fit()
        except Exception as e:
            forecast = [monthly_sales.mean()] * forecast_months
            forecast_df = pd.DataFrame({
                'Geographical Location': region,
                'Product Category': category,
                'Month of Purchase': range(latest_month + 1, latest_month + 1 + forecast_months),
                'Units Sold': forecast
            })
            forecast_results.append(forecast_df)
            continue

        forecast = fit.forecast(forecast_months)
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

    # Date processing
    df_full['Year of Purchase'] = pd.to_datetime(df_full['Timestamp Purchase']).dt.year
    df_full['Month of Purchase'] = pd.to_datetime(df_full['Timestamp Purchase']).dt.month
    df_full['Day Fraction'] = pd.to_datetime(df_full['Timestamp Purchase']).dt.day / pd.to_datetime(df_full['Timestamp Purchase']).dt.days_in_month

    monthly_sales_summary = df_full.groupby(
        ['Year of Purchase', 'Month of Purchase', 'Product Category', 'Geographical Location']
    ).agg({'Units Sold': 'sum', 'Day Fraction': 'mean'}).reset_index()

    global max_units_sold
    max_units_sold = monthly_sales_summary['Units Sold'].max()
    monthly_sales_summary['Bubble Size'] = (monthly_sales_summary['Units Sold'] / max_units_sold) * 60

    # Sidebar Filters
    regions = monthly_sales_summary['Geographical Location'].unique()
    custom_palette = ['#F5EEF8', '#E8DAEF', '#D2B4DE', '#BB8FCE', '#A569BD', '#884EA0', '#76448A']
    global color_map
    color_map = dict(zip(regions, custom_palette[:len(regions)]))

    # Move 'Choose Year for Visualization' Above Predictive Options
    available_years = sorted(df_full['Year of Purchase'].unique(), reverse=True)
    selected_graph_year = st.sidebar.selectbox("Choose Year for Visualization", available_years, index=0)

    st.sidebar.subheader("Predictive Analysis Options")
    show_predictive = st.sidebar.checkbox("Show Predictive Analysis", value=False)
    if show_predictive:
        forecast_months = st.sidebar.slider("Select Forecast Months", min_value=1, max_value=6, value=3)

    percentile_threshold = st.sidebar.slider("Select Percentile (%)", min_value=0, max_value=100, value=90)

    selected_years = st.sidebar.multiselect("Select Year(s)", available_years, default=available_years)
    selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=regions)
    selected_categories = st.sidebar.multiselect(
        "Select Product Category(ies)",
        monthly_sales_summary['Product Category'].unique(),
        default=monthly_sales_summary['Product Category'].unique()
    )

    # Filter data for graph and table
    threshold_value = monthly_sales_summary['Units Sold'].quantile(percentile_threshold / 100)
    filtered_graph_data = monthly_sales_summary[
        (monthly_sales_summary['Year of Purchase'] == selected_graph_year) &
        (monthly_sales_summary['Units Sold'] >= threshold_value) &
        (monthly_sales_summary['Geographical Location'].isin(selected_regions)) &
        (monthly_sales_summary['Product Category'].isin(selected_categories))
    ]

    # Forecast Data
    if show_predictive:
        forecast_df = forecast_sales_by_region_and_category(monthly_sales_summary, forecast_months=forecast_months)
        forecast_df['Bubble Size'] = (forecast_df['Units Sold'] / max_units_sold) * 60
        filtered_graph_data = pd.concat([filtered_graph_data, forecast_df], ignore_index=True)

    # Plot Function
    def plot_sales_with_forecast(data, year):
        fig = go.Figure()
        for region in regions:
            region_data = data[data['Geographical Location'] == region]
            historical_data = region_data[region_data['Year of Purchase'] == year]
            forecast_data = region_data[region_data['Year of Purchase'] > year]

            # Historical Data
            fig.add_trace(go.Scatter(
                x=historical_data['Month of Purchase'] + historical_data['Day Fraction'] - 1,
                y=historical_data['Product Category'],
                mode='markers',
                marker=dict(
                    size=(historical_data['Units Sold'] / max_units_sold) * 60,
                    color=color_map[region],
                    opacity=0.8
                ),
                name=f"{region} (Historical)"
            ))

            # Forecast Data
            if not forecast_data.empty:
                fig.add_trace(go.Scatter(
                    x=forecast_data['Month of Purchase'] + forecast_data['Day Fraction'] - 1,
                    y=forecast_data['Product Category'],
                    mode='markers',
                    marker=dict(
                        size=(forecast_data['Units Sold'] / max_units_sold) * 60,
                        color=color_map[region],
                        opacity=0.4
                    ),
                    name=f"{region} (Forecast)"
                ))

        fig.update_layout(
            title=f"Monthly Sales by Product Category for Year {year} (Predictive Included)" if show_predictive else f"Monthly Sales by Product Category for Year {year}",
            xaxis=dict(
                title="Month of Purchase",
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                range=[0, 12.5]
            ),
            yaxis=dict(
                title="Product Category",
                categoryorder="array",
                categoryarray=monthly_sales_summary['Product Category'].unique(),
                range=[-0.5, len(monthly_sales_summary['Product Category'].unique()) - 0.5]
            ),
            legend_title="Region",
            width=1200,
            height=600
        )
        return fig

    # Plot Graph
    if not filtered_graph_data.empty:
        graph_fig = plot_sales_with_forecast(filtered_graph_data, selected_graph_year)
        st.plotly_chart(graph_fig, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

    # Table: Average Units Sold
    st.subheader("Average Units Sold by Year, Region, and Product Category")
    avg_units_sold_summary = monthly_sales_summary.groupby(
        ['Year of Purchase', 'Geographical Location', 'Product Category']
    ).agg({'Units Sold': 'mean'}).reset_index()

    filtered_summary = avg_units_sold_summary[
        (avg_units_sold_summary['Year of Purchase'].isin(selected_years)) &
        (avg_units_sold_summary['Geographical Location'].isin(selected_regions)) &
        (avg_units_sold_summary['Product Category'].isin(selected_categories))
    ]

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

    # Create user segmentation for pie chart
    def segment_sessions(funnel_stages):
        if funnel_stages['Order Confirmation Page']:
            return 'Converted Users'
        elif funnel_stages['Cart Page']:
            return 'Engaged Users'
        else:
            return 'Abandoned Users'

    df_behavioral['Segment'] = df_behavioral['Funnel Stages'].apply(segment_sessions)
    segment_counts = df_behavioral['Segment'].value_counts()

    # Define custom purple-pink gradient palette
    custom_palette = [
        '#F5EEF8', '#E8DAEF', '#D2B4DE', '#BB8FCE', '#A569BD',
        '#884EA0', '#76448A', '#633974', '#512E5F', '#4A235A'
    ]

    # Plot the funnel analysis and pie chart side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot conversion rates as bars
    index = np.arange(len(funnel_stages))
    bar_width = 0.35
    bars = ax1.bar(index, conversion_rates, bar_width, label='Conversion Rate', color=custom_palette[3])

    # Plot drop-off rates as a line
    ax1.plot(index, dropoff_rates, label='Drop-Off Rate', color=custom_palette[5], marker='o', linestyle='-', linewidth=2)

    # Annotate conversion rates on bars
    for i, bar in enumerate(bars):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
            f'{conversion_rates[i]:.2f}%',
            ha='center', color='black', fontsize=10
        )

    # Annotate drop-off rates on line
    for i, rate in enumerate(dropoff_rates):
        ax1.text(
            index[i], rate + 2,
            f'{rate:.2f}%',
            ha='center', color='black', fontsize=10
        )

    # Style the funnel analysis plot
    ax1.set_xlabel('Funnel Stage', fontweight='bold')
    ax1.set_ylabel('Rate (%)', fontweight='bold')
    ax1.set_title('Conversion Rate with Drop-Off Rate Overlay', fontweight='bold')
    ax1.set_xticks(index)
    ax1.set_xticklabels(funnel_stages, rotation=45, ha='right')
    ax1.legend(loc='upper left')

    # Plot the pie chart
    colors = custom_palette[:3]  # Use the first three colors for the pie chart
    explode = (0.1, 0, 0)  # Emphasize Converted Users
    ax2.pie(
        segment_counts,
        labels=segment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        explode=explode,
        colors=colors
    )
    ax2.set_title('Session Segmentation by Funnel Progress', fontsize=14, fontweight='bold')

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


def inventory_analysis():
    st.header("Inventory Analysis")

    global df_inventory, df_item, df_transaction

    # Ensure datasets are properly loaded
    if df_inventory is None or df_item is None or df_transaction is None:
        st.error("Datasets are not loaded properly.")
        return

    # Merge 'Timestamp Purchase' from df_transaction into df_item
    df_transaction['Timestamp Purchase'] = pd.to_datetime(df_transaction['Timestamp Purchase'])
    df_item = pd.merge(df_item, df_transaction[['Transaction ID', 'Timestamp Purchase']], on='Transaction ID', how='left')

    # Convert 'Timestamp Purchase' to 'Date'
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

    # Initialize the Reorder Point column
    if 'Reorder Point' not in df_inventory.columns:
        df_inventory['Reorder Point'] = df_inventory['Average Daily Sales'] * df_inventory['Lead Time']

    # Sidebar: Graph Type Selection
    st.sidebar.subheader("Graph Type")
    graph_type = st.sidebar.radio("Select Graph Type", options=["Historical", "Predictive"])

    # Sidebar: Update Lead Time (Visible Only for Historical)
    if graph_type == "Historical":
        st.sidebar.subheader("Update Lead Time")
        selected_product = st.sidebar.selectbox(
            "Search and Select Product",
            options=df_inventory['Product ID'].unique(),
            format_func=lambda x: f"{x} - {df_inventory.loc[df_inventory['Product ID'] == x, 'Product Category'].iloc[0]}"
        )

        # Input box for Lead Time
        if selected_product:
            current_lead_time = int(df_inventory[df_inventory['Product ID'] == selected_product]['Lead Time'].iloc[0])
            new_lead_time = st.sidebar.number_input(
                f"Set Lead Time for Product {selected_product}",
                min_value=1,
                value=current_lead_time
            )

            # Update Lead Time and Reorder Point
            if new_lead_time != current_lead_time:
                df_inventory.loc[df_inventory['Product ID'] == selected_product, 'Lead Time'] = new_lead_time
                df_inventory.loc[df_inventory['Product ID'] == selected_product, 'Reorder Point'] = \
                    df_inventory.loc[df_inventory['Product ID'] == selected_product, 'Average Daily Sales'] * new_lead_time

    # Stock status calculation
    def determine_stock_status(row):
        if row['Current Stock'] >= row['Reorder Point']:
            return 'In Stock'
        elif 0 < row['Current Stock'] < row['Reorder Point']:
            return 'Low Stock'
        else:
            return 'Out of Stock'

    df_inventory['Stock Status'] = df_inventory.apply(determine_stock_status, axis=1)

    # Define custom purple-pink gradient palette
    custom_palette = [
        '#F5EEF8', '#E8DAEF', '#D2B4DE', '#BB8FCE', '#A569BD',
        '#884EA0', '#76448A', '#633974', '#512E5F', '#4A235A'
    ]

    fig, ax = plt.subplots(figsize=(20, 7))

    if graph_type == "Historical":
        bars = ax.bar(
            df_inventory['Product ID'],
            df_inventory['Current Stock'],
            color=[custom_palette[3] if status == 'In Stock' else custom_palette[6]
                   for status in df_inventory['Stock Status']],
            alpha=0.8,
            label='Current Stock'
        )
        ax.plot(
            df_inventory['Product ID'],
            df_inventory['Reorder Point'],
            color=custom_palette[8],
            marker='o',
            linewidth=1.5,
            label='Reorder Point'
        )
        ax.set_title('Inventory Levels and Reorder Points (Historical)', fontweight='bold', fontsize=14)

    elif graph_type == "Predictive":
        st.sidebar.subheader("Forecast Duration")
        forecast_days = st.sidebar.slider("Select Days for Forecast", min_value=7, max_value=60, value=30)

        # Generate forecasted sales
        forecasted_sales = {
            product_id: forecast_daily_sales(product_id, daily_sales, forecast_days)['Forecasted Sales'].sum()
            for product_id in df_inventory['Product ID']
        }
        df_inventory['Forecasted Sales'] = df_inventory['Product ID'].map(forecasted_sales)

        # Calculate additional units needed
        df_inventory['Additional Units Needed'] = df_inventory['Forecasted Sales'] - df_inventory['Current Stock']
        df_inventory['Additional Units Needed'] = df_inventory['Additional Units Needed'].apply(lambda x: max(x, 0))

        # Define color for bars
        bar_colors = [
            custom_palette[5] if row['Current Stock'] < row['Reorder Point'] else custom_palette[3]
            for _, row in df_inventory.iterrows()
        ]

        # Plot bars for current stock
        ax.bar(
            df_inventory['Product ID'],
            df_inventory['Current Stock'],
            color=bar_colors,
            alpha=0.8,
            label='Current Stock'
        )

        # Plot additional units needed
        ax.bar(
            df_inventory['Product ID'],
            df_inventory['Additional Units Needed'],
            bottom=df_inventory['Current Stock'],
            color=custom_palette[1],
            alpha=0.8,
            label='Forecasted Additional Units'
        )
        ax.set_title(f'Inventory Forecast for Next {forecast_days} Days', fontweight='bold', fontsize=14)

    ax.set_xlabel('Product ID', fontweight='bold', fontsize=12)
    ax.set_ylabel('Units', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 100)  # Ensure consistent scale
    ax.set_xticks(range(len(df_inventory['Product ID'])))
    ax.set_xticklabels(df_inventory['Product ID'], rotation=45, ha='right', fontsize=9)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Stock Status Summary and Product Category Summary side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stock Status")
        stock_status_summary = df_inventory.groupby('Stock Status').agg(Number_of_Types=('Product ID', 'nunique')).reset_index()
        st.table(stock_status_summary)

    with col2:
        st.subheader("Product Category")
        category_summary = df_inventory.groupby('Product Category').agg(
            Number_of_Product=('Product ID', 'nunique'),
            Average_Stock_Level=('Current Stock', 'mean')
        ).reset_index()
        st.table(category_summary)


def forecast_daily_sales(product_id, daily_sales, forecast_days=30):
    product_data = daily_sales[daily_sales['Product ID'] == product_id].set_index('Date')['Units Sold']
    if len(product_data) < 10:
        return pd.DataFrame({'Date': [], 'Forecasted Sales': []})

    model = SARIMAX(product_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=forecast_days)
    forecast_values = forecast.predicted_mean

    forecast_dates = pd.date_range(start=product_data.index.max() + timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({'Date': forecast_dates, 'Forecasted Sales': forecast_values})


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

    # Custom purple-pink gradient palette
    custom_palette = [
        '#F5EEF8', '#E8DAEF', '#D2B4DE', '#BB8FCE', '#A569BD',
        '#884EA0', '#76448A', '#633974', '#512E5F', '#4A235A'
    ]

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
            palette=custom_palette,  # Use the new purple-pink gradient palette
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
    avg_comparison.T.plot(kind='bar', ax=ax, rot=0, color=custom_palette[:len(avg_comparison.columns)])  # Use palette for consistency

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

