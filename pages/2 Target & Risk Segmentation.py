# Inside the "if page == 'Target':" block in app.py
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
st.title("🎯 Target & Risk Segmentation")

# =========================================================
#Sidebar Filters
#============================================================
df_filtered = apply_global_filters(df)

#===========================================================
# KPIs
#============================================================
total_defaults = df['TARGET'].sum()
default_rate = df['TARGET'].mean() * 100

# Group-wise default rates
def_rate_gender = df.groupby("CODE_GENDER")["TARGET"].mean() * 100
def_rate_edu = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean() * 100
def_rate_family = df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean() * 100
def_rate_housing = df.groupby("NAME_HOUSING_TYPE")["TARGET"].mean() * 100

# Averages for Defaulters
avg_income_def = df[df['TARGET'] == 1]['AMT_INCOME_TOTAL'].mean()
avg_credit_def = df[df['TARGET'] == 1]['AMT_CREDIT'].mean()
avg_annuity_def = df[df['TARGET'] == 1]['AMT_ANNUITY'].mean()
avg_emp_def = df[df['TARGET'] == 1]['EMPLOYMENT_YEARS'].mean()
          
col1, col2, col3 = st.columns(3)
col1.metric("Total Defaults", f"{total_defaults:,}")
col2.metric("Default Rate (%)", f"{default_rate:.2f}%")
col3.metric("Avg Income (Defaulters)", f"{avg_income_def:,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Credit (Defaulters)", f"{avg_credit_def:,.0f}")
col5.metric("Avg Annuity (Defaulters)", f"{avg_annuity_def:,.0f}")
col6.metric("Avg Employment Years (Defaulters)", f"{avg_emp_def:.1f}")

col7, col8, col9 = st.columns(3)
col7.metric("Default Rate by Gender (%)", f"{def_rate_gender.mean():.2f}%")
col8.metric("Default Rate by Education (%)", f"{def_rate_edu.mean():.2f}%")
col9.metric("Default Rate by Family Status (%)", f"{def_rate_family.mean():.2f}%")

st.metric("Default Rate by Housing Type (%)", f"{def_rate_housing.mean():.2f}%")

st.markdown("")

st.subheader("📈 Risk Segment Visuals")

#==================================================
# Graphs
#==================================================


# 1. Bar — Counts: Default vs Repaid
fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(x="TARGET", data=df)
plt.title("Counts: Default vs Repaid")
st.pyplot(fig)

# 2. Bar — Default % by Gender
fig, ax = plt.subplots(figsize=(8,5))
gender_default = df.groupby("CODE_GENDER")["TARGET"].mean()
gender_default.plot(kind="bar")
plt.title("Default % by Gender")
plt.ylabel("Default %")

# 3. Bar — Default % by Education
fig, ax = plt.subplots(figsize=(8,5))
edu_default = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean()
edu_default.plot(kind="bar")
plt.title("Default % by Education")
plt.ylabel("Default %")
st.pyplot(fig)

# Default % by Family Status
fig, ax = plt.subplots(figsize=(8,5))
fam_default = df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean()
fam_default.plot(kind="bar")
plt.title("Default % by Family Status")
plt.ylabel("Default %")
st.pyplot(fig)

# 5. Bar — Default % by Housing Type
fig, ax = plt.subplots(figsize=(8,5))
house_default = df.groupby("NAME_HOUSING_TYPE")["TARGET"].mean()
house_default.plot(kind="bar")
plt.title("Default % by Housing Type")
plt.ylabel("Default %")
st.pyplot(fig)

# 6. Boxplot — Income by Target
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x="TARGET", y="AMT_INCOME_TOTAL", data=df)
plt.title("Income by Target")
st.pyplot(fig)

# 7. Boxplot — Credit by Target
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x="TARGET", y="AMT_CREDIT", data=df)
plt.title("Credit by Target")
st.pyplot(fig)

# 8. Violin — Age vs Target
fig, ax = plt.subplots(figsize=(8,5))
sns.violinplot(x="TARGET", y="DAYS_BIRTH", data=df)
st.pyplot(fig)

# 9. Histogram (stacked) — EMPLOYMENT_YEARS by Target
fig, ax = plt.subplots(figsize=(8,5))
df["EMP_YEARS"] = (-df["DAYS_EMPLOYED"] / 365).astype(int)
sns.histplot(data=df, x="EMP_YEARS", hue="TARGET", multiple="stack")
plt.title("Employment Years by Target")
st.pyplot(fig)


# 10. Stacked Bar — Contract type vs Target
fig, ax = plt.subplots(figsize=(8,5))
contract_dist = df.groupby(["NAME_CONTRACT_TYPE", "TARGET"]).size().unstack(fill_value=0)
contract_dist.plot(kind="bar", stacked=True, ax=ax, color=["#3A993D", "#F44336"])
ax.set_ylabel("Count")
ax.set_xlabel("Contract Type")
st.pyplot(fig)
# -----------------------------
# Narrative Insights
# -----------------------------
st.subheader("📝 Insights")
st.markdown("""
-Employment Length: Applicants with shorter employment history (less than 2 years) tend to default more than those with longer job stability.
-Age: Younger applicants (under 30) have higher default rates compared to older age groups.
-Credit Amount: Higher loan amounts relative to income increase the likelihood of default.
-Contract Type: Clients with cash loans default more often than those with revolving credit or consumer credit.
-Income: Lower income applicants show a higher chance of default.
-Gender: Slightly higher default rates observed among females in the dataset.
-Family Status: Single or divorced applicants have higher default rates than married or widowed.
""")