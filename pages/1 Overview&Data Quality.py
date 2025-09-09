# Inside the "if page == 'Overview':" block in app.py
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
st.title("📊 Overview of Data Quality")

# =========================================================
#Sidebar Filters
#============================================================
df_filtered = apply_global_filters(df)

#===========================================================
# KPIs
#============================================================
total_applicants = df["SK_ID_CURR"].nunique()
default_rate = df["TARGET"].mean() * 100
repaid_rate = 100 - default_rate
total_features = df.shape[1]
avg_missing_per_feature = df.isnull().mean().mean() * 100
num_features = df.select_dtypes(include=[np.number]).shape[1]
cat_features = df.select_dtypes(exclude=[np.number]).shape[1]
median_age = df["AGE_YEARS"].median()
median_income = df["AMT_INCOME_TOTAL"].median()
avg_credit = df["AMT_CREDIT"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Applicants", f"{total_applicants:,}")
col2.metric("Default Rate (%)", f"{default_rate:.2f}%")
col3.metric("Repaid Rate (%)", f"{repaid_rate:.2f}%")

col4, col5, col6 = st.columns(3)
col4.metric("Total Features", total_features)
col5.metric("Num Features", num_features)
col6.metric("Cat Features", cat_features)

col7, col8, col9 = st.columns(3)
col7.metric("Avg Missing per Feature (%)", f"{avg_missing_per_feature:.2f}%")
col8.metric("Median Age (Years)", f"{median_age:.0f}")
col9.metric("Median Annual Income", f"{median_income:,.0f}")

st.metric("Average Credit Amount", f"{avg_credit:,.0f}")

st.markdown("")

st.subheader("📈 Data Distributions")

#==================================================
# Graphs
#==================================================

# 1.Pie / Donut — Target distribution (0 vs 1)
fig, ax = plt.subplots(figsize=(7,4))
df['TARGET'].value_counts().plot.pie(
    labels=['Repaid (0)', 'Default (1)'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['skyblue', 'salmon'],ax=ax
)
plt.title("Target Distribution")
plt.ylabel("")
st.pyplot(fig)

# 2. Bar — Top 20 features by missing %
missing = df.isnull().mean() * 100
top_missing = missing.sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=top_missing.values, y=top_missing.index, palette="viridis",ax=ax)
plt.xlabel("Missing %")
plt.title("Top 20 Features by Missing %")
st.pyplot(fig)

# 3. Histogram — AGE_YEARS
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df['AGE_YEARS'], bins=40, color="teal",ax=ax)
plt.xlabel("Age (Years)")
plt.title("Age Distribution")
st.pyplot(fig)

# 4. Histogram — AMT_INCOME_TOTAL
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df['AMT_INCOME_TOTAL'], bins=40, color="orange",ax=ax)
plt.xlabel("Annual Income")
plt.title("Income Distribution")
st.pyplot(fig)

# 5. Histogram — AMT_CREDIT
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df['AMT_CREDIT'], bins=40, color="purple",ax=ax)
plt.xlabel("Credit Amount")
plt.title("Credit Amount Distribution")
plt.show()
st.pyplot(fig)

# 6. Boxplot — AMT_INCOME_TOTAL
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(x=df['AMT_INCOME_TOTAL'], color="orange",ax=ax)
plt.xlabel("Annual Income")
plt.title("Boxplot: Income")
st.pyplot(fig)

# 7. Boxplot — AMT_CREDIT
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(x=df['AMT_CREDIT'], color="purple",ax=ax)
plt.xlabel("Credit Amount")
plt.title("Boxplot: Credit Amount")
st.pyplot(fig)

# 8. Countplot — CODE_GENDER
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(x='CODE_GENDER', data=df, palette="Set2")
plt.title("Gender Distribution")
st.pyplot(fig)

# 9. Countplot — NAME_FAMILY_STATUS
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(y='NAME_FAMILY_STATUS', data=df,
              order=df['NAME_FAMILY_STATUS'].value_counts().index,
              palette="Set1")
plt.title("Family Status Distribution")
st.pyplot(fig)

# 10. Countplot — NAME_EDUCATION_TYPE
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(y='NAME_EDUCATION_TYPE', data=df,
              order=df['NAME_EDUCATION_TYPE'].value_counts().index,
              palette="Set3")
plt.title("Education Type Distribution")
st.pyplot(fig)
#==================================================================================
# Narrative Insights
#==================================================================================
st.header("📝Insigths")
st.markdown("""               
The target distribution shows that most applicants successfully repay their loans, 
while defaults are a smaller portion, indicating an imbalanced dataset.

The income and credit variables are right-skewed, with a few applicants earning or borrowing 
extremely high amounts — potential outliers that may distort averages.

Some features exhibit high missing values (over 40–60%), which could pose risks 
to model reliability and may need dropping or careful imputation.
""")