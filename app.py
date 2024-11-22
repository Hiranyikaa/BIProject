import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter


<<<<<<< Updated upstream
# โหลด datasets เหมือนที่ทำใน Colab
df_behavioral = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Behavioral_Dataset.csv')
df_customer = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Customer_Dataset.csv')
df_item = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Transaction_Items_Dataset.csv')
df_transaction = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Transaction_Dataset.csv')
df_inventory = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Inventory_Dataset.csv')
df_campaign = pd.read_csv(r'C:\Users\UNS_CT\OneDrive\เอกสาร\PA6 dataset\PA6 dataset\Campaign_Dataset.csv')
=======
# Load datasets
df_behavioral = pd.read_csv(r'Behavioral_Dataset.csv')
df_customer = pd.read_csv(r'Customer_Dataset.csv')
df_item = pd.read_csv(r'Transaction_Items_Dataset.csv')
df_transaction = pd.read_csv(r'Transaction_Dataset.csv')
df_inventory = pd.read_csv(r'Inventory_Dataset.csv')
df_campaign = pd.read_csv(r'Campaign_Dataset.csv')
>>>>>>> Stashed changes

st.title("Data Overview")
st.write("Customer Dataset")
st.write(df_customer.head())

# ตัวอย่าง Heatmap
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
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt='.0f', ax=ax)
st.pyplot(fig)
