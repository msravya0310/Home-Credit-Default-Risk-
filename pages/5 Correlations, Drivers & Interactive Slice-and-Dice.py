# Inside the "if page == 'Correlations':" block in app.py
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
df=preprocess_data(df=load_data())
st.title("🔍 Correlations, Drivers & Interactive Slice-and-Dice")

# =========================================================
#Sidebar Filters
#============================================================
df_filtered = apply_global_filters(df)

#===========================================================================
#KPI'S
#============================================================================
corr_target = df.select_dtypes(include='number').corr()
target_corr = corr_target['TARGET'].drop('TARGET').sort_values(ascending=False)

top5_pos_corr = target_corr.head(5)

top5_neg_corr = target_corr.tail(5).sort_values()

income_corr = corr_target['AMT_INCOME_TOTAL'].drop('AMT_INCOME_TOTAL').abs()
most_corr_income = income_corr.idxmax(), income_corr.max()
col_name, corr_value = most_corr_income

credit_corr =corr_target['AMT_CREDIT'].drop('AMT_CREDIT').abs()
most_corr_credit = credit_corr.idxmax(), credit_corr.max()
col_name, corr_value = most_corr_credit


corr_income_credit = corr_target.loc['AMT_INCOME_TOTAL', 'AMT_CREDIT']

corr_age_target = corr_target.loc['AGE_YEARS', 'TARGET']

corr_emp_target = corr_target.loc['EMPLOYMENT_YEARS', 'TARGET']

corr_fam_target =corr_target.loc['CNT_FAM_MEMBERS', 'TARGET']

top5_features = target_corr.abs().sort_values(ascending=False).head(5)
variance_explained_proxy = top5_features.sum()

num_features_high_corr = (target_corr.abs() > 0.5).sum()

col1, col2 = st.columns(2)
with col1:
    st.metric("Top 5 +Corr (TARGET)", ", ".join([f"{x} ({y:.2f})" for x,y in top5_pos_corr.items()]))
    st.metric("Top 5 −Corr (TARGET)", ", ".join([f"{x} ({y:.2f})" for x,y in top5_neg_corr.items()]))
    st.metric("Most correlated with Income", f"{col_name} ({corr_value:.2f})")
    st.metric("Most correlated with Credit", f"{col_name} ({corr_value:.2f})")
    st.metric("Corr(Income, Credit)", f"{corr_income_credit:.2f}")

with col2:
    st.metric("Corr(Age, TARGET)", f"{corr_age_target:.2f}")
    st.metric("Corr(EmploymentY, TARGET)", f"{corr_emp_target:.2f}")
    st.metric("Corr(Family Size, TARGET)", f"{corr_fam_target:.2f}")
    st.metric("Variance Explained (Top 5)", f"{variance_explained_proxy:.2f}")
    st.metric("# Features |corr| > 0.5", num_features_high_corr)
st.markdown("")

st.subheader("📈 Correlation & Drivers Visuals")

#===================================================
#Graphs
#===================================================

col1,col2 = st.columns(2)
with col1:
    #  Heatmap — Correlation (selected numerics)
    fig, ax = plt.subplots(figsize=(8,5))
    selected = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 'CNT_FAM_MEMBERS', 'TARGET']
    sns.heatmap(df[selected].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (Selected Numerics)')
    st.pyplot(fig)
 
with col2:
    #  2. Bar — |Correlation| of features vs TARGET (top N)
    fig, ax = plt.subplots(figsize=(8,5))
    corrs = df[selected].corr()['TARGET'].drop('TARGET').abs().sort_values(ascending=False)
    corrs.head(5).plot(kind='bar')
    plt.title('|Correlation| with TARGET (Top 5)')
    plt.ylabel('Absolute Correlation')
    st.pyplot(fig)


col3,col4 = st.columns(2)
with col3:
    # Scatter — Age vs Credit (hue=TARGET) 
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='AGE_YEARS', y='AMT_CREDIT', hue='TARGET', data=df, alpha=0.5)
    plt.title('AGE vs Credit (hue=TARGET)')
    st.pyplot(fig)

with col4:
    #  Scatter — Age vs Income (hue=TARGET) 
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='AGE_YEARS', y='AMT_INCOME_TOTAL', hue='TARGET', data=df, alpha=0.5)
    plt.title('Age vs Income (hue=TARGET)')
    st.pyplot(fig)

col5,col6 = st.columns(2)
with col5:
    #  Scatter — Employment Years vs TARGET (jitter) 
    fig, ax = plt.subplots(figsize=(8,5))
    plt.scatter(df['EMPLOYMENT_YEARS'], df['TARGET'], s=10, alpha=0.5)
    plt.title('Employment Years vs TARGET')
    plt.xlabel('Years Employed')
    plt.ylabel('TARGET')
    plt.yticks([0, 1])
    st.pyplot(fig)

with col6:
    #  Boxplot — Credit by Education 
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x='NAME_EDUCATION_TYPE', y='AMT_CREDIT', data=df)
    plt.title('Credit Amount by Education')
    plt.xticks(rotation=30)
    st.pyplot(fig)

col7,col8 = st.columns(2)
with col7:
    #  Boxplot — Income by Family Status 
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x='NAME_FAMILY_STATUS', y='AMT_INCOME_TOTAL', data=df)
    plt.title('Income by Family Status')
    plt.xticks(rotation=30)
    st.pyplot(fig)

with col8:
    #  Pair Plot — Income, Credit, Annuity, TARGET 
    fig, ax = plt.subplots(figsize=(8,5))
    pairplot = sns.pairplot(df[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'TARGET']], hue='TARGET')
    pairplot.fig.suptitle('Pair Plot — Income, Credit, Annuity, TARGET', y=1.02)
    st.pyplot(pairplot.fig)

col9,col10 = st.columns(2)
with col9:
    #  Bar — Default Rate by Gender 
    fig, ax = plt.subplots(figsize=(8,5))
    default_by_gender = df.groupby('CODE_GENDER')['TARGET'].mean()
    default_by_gender.plot(kind='bar')
    plt.title('Default Rate by Gender')
    st.pyplot(fig)

with col10:
    #  Bar — Default Rate by Education 
    fig, ax = plt.subplots(figsize=(8,5))
    default_by_edu = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean()
    default_by_edu.plot(kind='bar')
    plt.title('Default Rate by Education')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=30)
    st.pyplot(fig)

#=========================================================
# Narrative Insights
#=========================================================
st.subheader("📝 Insights")
st.markdown("""
-Income and Credit Relationship: Higher income generally correlates with higher credit amounts, 
indicating that clients with larger incomes are approved for bigger loans.

-Default Risk Varies by Income and Gender: Default rates tend to be higher in lower income brackets, 
highlighting affordability issues. Additionally, differences in default rates between genders 
(or other demographic groups)can point to specific risk profiles that should be considered in credit risk modeling.

-Employment Length Influences Default Probability:Clients with longer employment histories show lower default rates, likely due to increased financial stability.
his insight supports incorporating employment duration as a key feature in predictive models.
""")