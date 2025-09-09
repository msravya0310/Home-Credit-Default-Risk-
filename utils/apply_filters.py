import streamlit as st

def apply_global_filters(df):
    st.sidebar.header("🔍 Global Filters")


    # --- Gender ---
    gender = st.sidebar.multiselect(
        "Gender",
        df["CODE_GENDER"].dropna().unique().tolist(),
        default=df["CODE_GENDER"].dropna().unique().tolist()
    )

    # --- Education ---
    education = st.sidebar.multiselect(
        "Education",
        df["NAME_EDUCATION_TYPE"].dropna().unique().tolist(),
        default=df["NAME_EDUCATION_TYPE"].dropna().unique().tolist()
    )

    # --- Family Status ---
    family_status = st.sidebar.multiselect(
        "Family Status",
        df["NAME_FAMILY_STATUS"].dropna().unique().tolist(),
        default=df["NAME_FAMILY_STATUS"].dropna().unique().tolist()
    )

    # --- Housing Type ---
    housing = st.sidebar.multiselect(
        "Housing Type",
        df["NAME_HOUSING_TYPE"].dropna().unique().tolist(),
        default=df["NAME_HOUSING_TYPE"].dropna().unique().tolist()
    )

    # --- Age Range (converted from DAYS_BIRTH) ---
    min_age = int(df["DAYS_BIRTH"].apply(lambda x: -x / 365).min())
    max_age = int(df["DAYS_BIRTH"].apply(lambda x: -x / 365).max())
    age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

    # --- Income Range ---
    min_income = int(df["AMT_INCOME_TOTAL"].min())
    max_income = int(df["AMT_INCOME_TOTAL"].max())
    income_range = st.sidebar.slider("Income Bracket", min_income, max_income, (min_income, max_income), step=10000)

  # Save in session_state
    st.session_state["filters"] = {
        "gender": gender,
        "education": education,
        "family_status": family_status,
        "housing": housing,
        "age_range": age_range,
        "income_range": income_range,
    }

    # --- Apply Filters ---
    df_filtered = df[
        df["CODE_GENDER"].isin(gender) &
        df["NAME_EDUCATION_TYPE"].isin(education) &
        df["NAME_FAMILY_STATUS"].isin(family_status) &
        df["NAME_HOUSING_TYPE"].isin(housing) &
        df["DAYS_BIRTH"].apply(lambda x: -x / 365).between(age_range[0], age_range[1]) &
        df["AMT_INCOME_TOTAL"].between(income_range[0], income_range[1])
    ]
    return  df_filtered
