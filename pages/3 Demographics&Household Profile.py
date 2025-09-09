# Inside the "if page == 'Demographics':" block in app.py
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
st.title("🏠 Demographics & Household Profile")

# =========================================================
#Sidebar Filters
#============================================================
df_filtered = apply_global_filters(df)

#===========================================================
# KPIs
#============================================================
male_pct = (df['CODE_GENDER'].str.lower().eq('male').mean()) * 100
female_pct = (df['CODE_GENDER'].str.lower().eq('female').mean()) * 100
avg_age_def = df.loc[df["TARGET"] == 1, "AGE_YEARS"].mean()
avg_age_nondef = df.loc[df["TARGET"] == 0, "AGE_YEARS"].mean()
pct_with_children = (df['CNT_CHILDREN'] > 0).mean() * 100
avg_family_size = df['CNT_FAM_MEMBERS'].mean()
avg_family_size = df['CNT_FAM_MEMBERS'].mean()
pct_married = df['NAME_FAMILY_STATUS'].str.lower().eq('married').mean() * 100
pct_single = df['NAME_FAMILY_STATUS'].str.lower().eq('single').mean() * 100
higher_edu = ['Bachelor', 'Master', 'PhD']
pct_higher_edu = df['NAME_EDUCATION_TYPE'].isin(higher_edu).mean() * 100
pct_with_parents = df['NAME_HOUSING_TYPE'].eq('With parents').mean() * 100
pct_working = df['OCCUPATION_TYPE'].notna().mean() * 100  
avg_employment_years = df['EMPLOYMENT_YEARS'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("% Male ", f"{male_pct:.1f}")
col2.metric("% Female", f"{female_pct:.1f}%")
col3.metric("Avg Age — Defaulters ", f"{avg_age_def:,.1f}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Age — Non-Defaulters", f"{avg_age_nondef :,.1f}")
col5.metric("% With Children", f"{pct_with_children:,.1f}")
col6.metric("Avg Family Size ", f"{avg_family_size:.1f}")

col7, col8, col9 = st.columns(3)
col7.metric("% Married ", f"{pct_married:.1f}%")
col8.metric("% Single ", f"{pct_single:.1f}%")
col9.metric("% Higher Education ", f"{pct_higher_edu:.1f}%")

col10 = st.columns(1)[0]
col10.metric("% Currently Working ", f"{pct_working:.1f}%")
st.metric("Avg Employment Years", f"{avg_employment_years:.1f}%")

st.markdown("")

st.subheader("📈 Demographic Visuals")


#==================================================
# Graphs
#==================================================

# 1. Histogram — Age distribution (all)
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(df['AGE_YEARS'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('AGE_YEARS')
plt.ylabel('Count')
st.pyplot(fig)

#2. Histogram — Age by Target (overlay)
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(data=df, x='AGE_YEARS', hue='TARGET', bins=30, kde=True, alpha=0.5)
plt.title('Age Distribution by Target')
plt.xlabel('AGE_YEARS')
plt.ylabel('Count')
st.pyplot(fig)

# 3. Bar — Gender distribution.6
fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(x='CODE_GENDER', data=df)
plt.title('Gender Distribution')
plt.xlabel('CODE_GENDER')
plt.ylabel('Count')
st.pyplot(fig)

# 4. Bar — Family Status distribution
fig, ax = plt.subplots(figsize=(7,5))
sns.countplot(x='NAME_FAMILY_STATUS', data=df)
plt.title('Family Status Distribution')
plt.xlabel('NAME_FAMILY_STATUS')
plt.ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)

# 5. Bar — Education distribution
fig, ax = plt.subplots(figsize=(7,5))
sns.countplot(x='NAME_EDUCATION_TYPE', data=df)
plt.title('Education Distribution')
plt.xlabel('NAME_EDUCATION_TYPE')
plt.ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)

# 6. Bar — Occupation distribution (top 10)
fig, ax = plt.subplots(figsize=(9,5))
top_occupations = df['OCCUPATION_TYPE'].value_counts().nlargest(10)
sns.barplot(x=top_occupations.index, y=top_occupations.values)
plt.title('Top 10 Occupations')
plt.xlabel('OCCUPATION_TYPE')
plt.ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)

# 7. Pie — Housing Type distribution
fig, ax = plt.subplots(figsize=(8,6))
df['NAME_HOUSING_TYPE'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Housing Type Distribution')
plt.xlabel("NAME_HOUSING_TYPE")
plt.ylabel('')
st.pyplot(fig)

# 8. Countplot — CNT_CHILDREN
fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(x='CNT_CHILDREN', data=df)
plt.title('Number of Children')
plt.xlabel('CNT_CHILDREN')
plt.ylabel('Count')
st.pyplot(fig)

# 9. Boxplot — Age vs Target
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='TARGET', y='AGE_YEARS', data=df)
plt.title('Age vs Target')
plt.xlabel('Target')
plt.ylabel('AGE_YEARS')
st.pyplot(fig)

# 10. Heatmap — Corr(Age, Children, Family Size, TARGET)
fig, ax = plt.subplots(figsize=(8,6))
cols = ['AGE_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'TARGET']
corr = df[cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
st.pyplot(fig)

# -----------------------------
# Narrative Insights
# -----------------------------
st.subheader("📝 Insights")
st.markdown("""
-These life-stage patterns suggest that credit risk is not determined by age or family size alone, but by their intersection. 
-Younger individuals—especially those supporting larger families—are at higher risk, while older, stable family units are less likely to default. 
-Lenders may consider targeted support, education, or stricter risk assessment for applicants 
in early life stages with greater household responsibilities.
""")