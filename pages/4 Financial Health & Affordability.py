import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.load_data import load_data
from preprocess import preprocess_data
from utils.apply_filters import apply_global_filters


#=====================================================
#Loadset
#=======================================================
df = preprocess_data(df=load_data())
# Add DTI and LTI to dataframe
df['DTI'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['LTI'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

st.title("💰 Financial Health & Affordability")

# =========================================================
#Sidebar Filters
#============================================================
df_filtered = apply_global_filters(df)


#=========================================================
# KPIs
#=========================================================
avg_income = df['AMT_INCOME_TOTAL'].mean()
median_income = df['AMT_INCOME_TOTAL'].median()
avg_credit = df['AMT_CREDIT'].mean()
avg_annuity = df['AMT_ANNUITY'].mean()
avg_goods_price = df['AMT_GOODS_PRICE'].mean()
avg_dti = df['DTI'].mean()
avg_lti = df['LTI'].mean()
income_non_def = df.loc[df['TARGET'] == 0, 'AMT_INCOME_TOTAL'].mean()
income_def = df.loc[df['TARGET'] == 1, 'AMT_INCOME_TOTAL'].mean()
income_gap = income_non_def - income_def
credit_non_def = df.loc[df['TARGET'] == 0, 'AMT_CREDIT'].mean()
credit_def = df.loc[df['TARGET'] == 1, 'AMT_CREDIT'].mean()
credit_gap = credit_non_def - credit_def
pct_high_credit = (df['AMT_CREDIT'] > 1_000_000).mean() * 100

# Display KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Avg Annual Income", f"{avg_income:,.0f}")
col2.metric("Median Annual Income", f"{median_income:,.0f}")
col3.metric("Avg Credit Amount", f"{avg_credit:,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Annuity", f"{avg_annuity:,.0f}")
col5.metric("Avg Goods Price", f"{avg_goods_price:,.0f}")
col6.metric("Debt-to-Income Ratio (DTI)", f"{avg_dti:.2f}", help="AMT_ANNUITY / AMT_INCOME_TOTAL")

col7, col8, col9 = st.columns(3)
col7.metric("Loan-to-Income Ratio (LTI)", f"{avg_lti:.2f}", help="AMT_CREDIT / AMT_INCOME_TOTAL")
col8.metric("Income Gap (Non-def − Def)", f"{income_gap:,.0f}")
col9.metric("Credit Gap (Non-def − Def)", f"{credit_gap:,.0f}")

col10 = st.columns(1)[0]
col10.metric("% High Credit (>1M)", f"{pct_high_credit:.1f}%")

st.markdown("")

#=========================================================
# Visualizations
#=========================================================
st.subheader("💰 Financial Visuals")

#=========================================================
#graphs
#==========================================================

col1,col2 = st.columns(2)
with col1:
    #  Histogram — Income distribution
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(df['AMT_INCOME_TOTAL'], bins=30, kde=True)
    plt.title('Income Distribution')
    plt.xlabel('Annual Income')
    plt.ylabel('Count')
    st.pyplot(fig)

with col2:
    # Histogram — Credit distribution
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(df['AMT_CREDIT'], bins=30, kde=True)
    plt.title('Credit Amount Distribution')
    plt.xlabel('Credit Amount')
    plt.ylabel('Count')
    st.pyplot(fig)

col3,col4 = st.columns(2)
with col3:
    #  Histogram — Annuity distribution
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(df['AMT_ANNUITY'], bins=30, kde=True)
    plt.title('Annuity Distribution')
    plt.xlabel('Annuity Amount')
    plt.ylabel('Count')
    st.pyplot(fig)

with col4:
    
    # Scatter — Income vs Credit
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_CREDIT', data=df, alpha=0.3)
    plt.title('Income vs Credit Amount')
    plt.xlabel('Annual Income')
    plt.ylabel('Credit Amount')
    st.pyplot(fig)

col5,col6 = st.columns(2)
with col5:
    #  Scatter — Income vs Annuity
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_ANNUITY', data=df, alpha=0.3)
    plt.title('Income vs Annuity')
    plt.xlabel('Annual Income')
    plt.ylabel('Annuity Amount')
    st.pyplot(fig)

with col6:
    # Boxplot — Credit by Target
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x='TARGET', y='AMT_CREDIT', data=df)
    plt.title('Credit Amount by Default Status')
    plt.xlabel('Target (Default)')
    plt.ylabel('Credit Amount')
    st.pyplot(fig)
    
col7,col8 = st.columns(2)
with col7:
    # Boxplot — Income by Target
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df)
    plt.title('Income by Default Status')
    plt.xlabel('Target (Default)')
    plt.ylabel('Annual Income')
    st.pyplot(fig)

with col8:
    # KDE / Density — Joint Income–Credit
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
            x=df['AMT_INCOME_TOTAL'], 
            y=df['AMT_CREDIT'],
            alpha=0.3, 
            s=10
            )
    plt.title('Scatterplot of Income vs Credit')
    plt.xlabel('Annual Income')
    plt.ylabel('Credit Amount')
    st.pyplot(fig)


col9,col10 = st.columns(2)
with col9:
    # Bar — Income Brackets vs Default Rate
    fig, ax = plt.subplots(figsize=(10,8))
    bins = [0, 100000, 200000, 400000, 600000, 1_000_000, np.inf]
    labels = ['<100K', '100K-200K', '200K-400K', '400K-600K', '600K-1M', '>1M']
    df['Income Bracket'] = pd.cut(df['AMT_INCOME_TOTAL'], bins=bins, labels=labels)
    default_rate = df.groupby('Income Bracket')['TARGET'].mean()
    default_rate.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    plt.title('Default Rate by Income Bracket')
    plt.xlabel('Income Bracket')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col10:
    #  Heatmap — Correlation of Financial Variables
    fig, ax = plt.subplots(figsize=(10,8))
    corr_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DTI', 'LTI', 'TARGET']
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap - Financial Variables')
    st.pyplot(fig)

#=========================================================
# Narrative Insights
#=========================================================
st.subheader("📝 Insights")
st.markdown("""
- **Default risk increases sharply** when affordability thresholds are breached:
    - **LTI > 6**: Indicates borrowers may be over-leveraged.
    - **DTI > 0.35**: Suggests limited disposable income after debt obligations.
- These thresholds are validated through binning and statistical modeling (e.g., logistic regression).
- Helps define **risk boundaries** for lending decisions and credit policy.
""")
