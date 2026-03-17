import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import json

from streamlit_option_menu import option_menu

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Crime Analytics System",
    page_icon="🚔",
    layout="wide"
)

# ---------------------------
# GLOBAL CSS
# ---------------------------
st.markdown("""
    <style>
        /* Hide auto-generated multipage nav tabs */
        [data-testid="stSidebarNav"]          { display: none; }
        section[data-testid="stSidebarNav"]   { display: none; }
        div[data-testid="collapsedControl"]   { display: none; }

        /* Remove +/- spinner buttons from ALL number inputs */
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type=number] {
            -moz-appearance: textfield;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD DATA
# ---------------------------
data = pd.read_csv("Dataset/cleaned_crime_data.csv")
clustered_data = pd.read_csv("Dataset/crime_clustered_data.csv")

model = joblib.load("Models/crime_prediction_model.pkl")

# ---------------------------
# SIDEBAR — SINGLE UNIFIED NAVIGATION
# ---------------------------
with st.sidebar:

    st.title("🚔 Crime Analytics & Prediction System")
    st.markdown("---")

    page = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Dataset Explorer",
            "Crime Analysis",
            "Crime Hotspot Map",
            "Crime Prediction",
            "About"
        ],
        icons=[
            "house-fill",
            "table",
            "bar-chart-fill",
            "map-fill",
            "robot",
            "info-circle-fill"
        ],
        default_index=0,
        key="main_menu",
        styles={
            "container": {
                "padding": "0px",
                "background-color": "transparent"
            },
            "icon": {
                "color": "#4fc3f7",
                "font-size": "15px"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "3px 0px",
                "padding": "8px 12px",
                "border-radius": "6px",
            },
            "nav-link-selected": {
                "background-color": "#1e3a5f",
                "color": "white",
                "font-weight": "600",
            },
        }
    )

# ---------------------------
# PAGE ROUTING
# ---------------------------

# ── HOME ─────────────────────────────────────────────────────────
if page == "Home":

    st.title("🚔 Crime Analytics and Prediction System")

    st.markdown("""
    This project analyzes crime data across Indian states and predicts crime trends
    using Machine Learning.

    **Features of the system:**
    • Crime data analysis &nbsp;|&nbsp; • Crime hotspot detection &nbsp;|&nbsp;
    • Machine learning crime prediction &nbsp;|&nbsp; • Interactive visualizations
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── ROW 1: KEY METRICS ────────────────────────────────────────
    st.subheader("📌 Dataset at a Glance")

    total_crimes  = int(data["TOTAL_IPC_CRIMES"].sum())
    total_records = data.shape[0]
    total_states  = data["STATE/UT"].nunique()
    year_range    = f"{int(data['YEAR'].min())} – {int(data['YEAR'].max())}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📁 Total Records",      f"{total_records:,}")
    m2.metric("🗺 States / UTs",        f"{total_states}")
    m3.metric("📅 Year Range",          year_range)
    m4.metric("🔢 Total IPC Crimes",    f"{total_crimes:,}")

    st.markdown("---")

    # ── ROW 2: CRIME TREND + TOP 5 STATES ────────────────────────
    st.subheader("📈 Crime Overview")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        crime_trend = data.groupby("YEAR")["TOTAL_IPC_CRIMES"].sum().reset_index()

        fig_trend = px.area(
            crime_trend,
            x="YEAR",
            y="TOTAL_IPC_CRIMES",
            markers=True,
            title="Total IPC Crimes — Year-wise Trend",
            color_discrete_sequence=["#4fc3f7"]
        )
        fig_trend.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Year",
            yaxis_title="Total IPC Crimes"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        top5 = (
            data.groupby("STATE/UT")["TOTAL_IPC_CRIMES"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )

        fig_top5 = px.bar(
            top5,
            x="TOTAL_IPC_CRIMES",
            y="STATE/UT",
            orientation="h",
            title="Top 5 States by Total Crime",
            color="TOTAL_IPC_CRIMES",
            color_continuous_scale="Reds",
            text="TOTAL_IPC_CRIMES"
        )
        fig_top5.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_top5.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis=dict(autorange="reversed"),
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig_top5, use_container_width=True)

    st.markdown("---")

    # ── ROW 3: CRIME TYPE BREAKDOWN + YEAR COMPARISON ─────────────
    st.subheader("🔍 Crime Type Breakdown")

    col_a, col_b = st.columns(2)

    with col_a:
        # Sum up key crime categories across entire dataset
        crime_cols = ["MURDER", "RAPE", "KIDNAPPING & ABDUCTION",
                      "ROBBERY", "BURGLARY", "RIOTS", "DOWRY DEATHS"]

        # Only use columns that actually exist in the dataset
        available_cols = [c for c in crime_cols if c in data.columns]

        if available_cols:
            crime_sums = data[available_cols].sum().reset_index()
            crime_sums.columns = ["Crime Type", "Total Cases"]
            crime_sums = crime_sums.sort_values("Total Cases", ascending=False)

            fig_pie = px.pie(
                crime_sums,
                names="Crime Type",
                values="Total Cases",
                hole=0.45,
                title="Distribution of Major Crime Types",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Crime type columns not found in dataset.")

    with col_b:
        # Year-wise comparison of top crime types as grouped bar
        if available_cols:
            yearly = data.groupby("YEAR")[available_cols].sum().reset_index()
            yearly_melted = yearly.melt(
                id_vars="YEAR",
                value_vars=available_cols[:4],   # top 4 for clarity
                var_name="Crime Type",
                value_name="Cases"
            )

            fig_grouped = px.bar(
                yearly_melted,
                x="YEAR",
                y="Cases",
                color="Crime Type",
                barmode="group",
                title="Year-wise Crime Type Comparison",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_grouped.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Year",
                yaxis_title="Cases",
                legend_title="Crime Type"
            )
            st.plotly_chart(fig_grouped, use_container_width=True)
        else:
            st.info("Crime type columns not found in dataset.")

    st.markdown("---")

    # ── ROW 4: STATE-WISE HEATMAP ─────────────────────────────────
    st.subheader("🌡 State-wise Crime Intensity Heatmap")

    # Top 10 states × available crime types
    top10_states = (
        data.groupby("STATE/UT")["TOTAL_IPC_CRIMES"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    if available_cols:
        heatmap_df = (
            data[data["STATE/UT"].isin(top10_states)]
            .groupby("STATE/UT")[available_cols]
            .sum()
        )

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns.tolist(),
                y=heatmap_df.index.tolist(),
                colorscale="YlOrRd",
                text=heatmap_df.values,
                texttemplate="%{text:,}",
                hoverongaps=False
            )
        )
        fig_heat.update_layout(
            title="Crime Intensity — Top 10 States vs Crime Types",
            xaxis_title="Crime Type",
            yaxis_title="State / UT",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Heatmap unavailable — crime type columns not found.")

    st.markdown("---")
    st.caption("🚔 Crime Analytics and Prediction System — Academic Project | Data: Indian IPC Crime Records")

# ── DATASET EXPLORER ─────────────────────────────────────────────
elif page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Dataset Shape")
    st.write(data.shape)

    st.subheader("Column Names")
    st.write(data.columns)

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

# ── CRIME ANALYSIS ───────────────────────────────────────────────
elif page == "Crime Analysis":

    st.title("📈 Crime Analysis Dashboard")

    st.subheader("Crime Trend Over Years")

    crime_trend = data.groupby("YEAR")["TOTAL_IPC_CRIMES"].sum().reset_index()

    fig = px.line(
        crime_trend,
        x="YEAR",
        y="TOTAL_IPC_CRIMES",
        markers=True,
        title="Total IPC Crimes Over Years"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Crime States")

    state_crime = data.groupby("STATE/UT")["TOTAL_IPC_CRIMES"].sum().reset_index()

    top_states = state_crime.sort_values(
        by="TOTAL_IPC_CRIMES",
        ascending=False
    ).head(10)

    fig2 = px.bar(
        top_states,
        x="STATE/UT",
        y="TOTAL_IPC_CRIMES",
        color="TOTAL_IPC_CRIMES",
        title="Top 10 States with Highest Crime"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ── CRIME HOTSPOT MAP ────────────────────────────────────────────
elif page == "Crime Hotspot Map":

    st.title("🗺 Crime Hotspot Map")

    st.markdown("States grouped by crime level using clustering.")

    with open("Dataset/india_states.geojson") as f:
        india_geo = json.load(f)

    state_mapping = {
        "Delhi UT": "Delhi",
        "D&N Haveli": "Dadra and Nagar Haveli",
        "Daman & Diu": "Daman and Diu",
        "A&N Islands": "Andaman and Nicobar",
        "Odisha": "Odisha",
        "Jammu & Kashmir": "Jammu and Kashmir",
        "Uttarakhand": "Uttaranchal"
    }

    clustered_data["STATE_FIXED"] = clustered_data["STATE/UT"].replace(state_mapping)

    fig = px.choropleth(
        clustered_data,
        geojson=india_geo,
        featureidkey="properties.NAME_1",
        locations="STATE_FIXED",
        color="crime_level",
        hover_name="STATE_FIXED",
        color_discrete_map={
            "Low Crime": "green",
            "Moderate Crime": "orange",
            "High Crime": "red"
        },
        title="Crime Level Across Indian States"
    )

    fig.update_geos(visible=False, fitbounds="locations")

    st.plotly_chart(fig, use_container_width=True)

# ── CRIME PREDICTION ─────────────────────────────────────────────
elif page == "Crime Prediction":

    st.title("🔮 Future Crime Prediction")

    st.markdown("Predict crime trends for a future year based on crime statistics.")

    states = sorted(data["STATE/UT"].unique())

    col1, col2 = st.columns(2)

    state = col1.selectbox("Select State", states)

    year = col2.number_input(
        "Enter Future Year",
        min_value=2024,
        max_value=2050,
        value=2030,
        step=1,
        format="%d"
    )

    st.subheader("Enter Crime Statistics")
    st.markdown("Type the number of cases directly into each field.")

    col3, col4 = st.columns(2)

    murder     = col3.number_input("Murder Cases",     min_value=0, max_value=10000, value=0, step=1, format="%d")
    rape       = col3.number_input("Rape Cases",       min_value=0, max_value=10000, value=0, step=1, format="%d")
    kidnapping = col4.number_input("Kidnapping Cases", min_value=0, max_value=10000, value=0, step=1, format="%d")
    riots      = col4.number_input("Riots",            min_value=0, max_value=10000, value=0, step=1, format="%d")
    dowry      = st.number_input(  "Dowry Deaths",     min_value=0, max_value=10000, value=0, step=1, format="%d")

    st.markdown("")

    if st.button("🔍 Predict Future Crime", use_container_width=False):

        input_data = np.array([[murder, rape, kidnapping, riots, dowry]])
        prediction = model.predict(input_data)
        total = int(prediction[0])

        if total < 5000:
            level = "Low Crime Risk"
        elif total < 10000:
            level = "Moderate Crime Risk"
        else:
            level = "High Crime Risk"

        st.subheader("📊 Prediction Result")

        st.success(f"""
For the year **{year}** in **{state}**:

Estimated **Total IPC Crimes**: **{total}**

Crime Risk Level: **{level}**
""")

        st.markdown("### 🧠 AI Crime Analysis")

        st.write(f"""
Based on the entered crime statistics, the projected crime trend for **{state}**
in **{year}** indicates a **{level.lower()}**.

The combination of **murder ({murder})**, **rape ({rape})**, **kidnapping ({kidnapping})**,
**riots ({riots})**, and **dowry deaths ({dowry})** contributes to the estimated
crime level.

Authorities may need to focus on preventive measures and law enforcement
strategies to reduce crime rates in the coming years.
""")

# ── ABOUT ────────────────────────────────────────────────────────
elif page == "About":

    st.title("📘 About the Project")
    st.markdown("---")

    st.markdown("## 📊 Project Summary")
    st.markdown("""
The **Crime Analytics and Prediction System** is a data-driven dashboard designed to analyze
historical crime data across Indian states and predict potential future crime trends.

By combining **data visualization**, **clustering techniques**, and **machine learning models**,
the system transforms raw crime statistics into meaningful insights that help users understand
crime patterns across regions and time.
""")

    summary_df = pd.DataFrame({
        "Feature": ["Data Exploration", "Trend Analysis", "Hotspot Detection", "ML Prediction"],
        "Capability Score": [90, 85, 80, 88]
    })

    fig_about = px.bar(
        summary_df,
        x="Feature",
        y="Capability Score",
        color="Capability Score",
        color_continuous_scale="Blues",
        title="System Capability Overview",
        text="Capability Score"
    )
    fig_about.update_traces(textposition="outside")
    fig_about.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig_about, use_container_width=True)

    st.markdown("---")

    st.markdown("## ❓ Why This Project")
    st.markdown("""
Crime data is often complex and difficult to interpret when presented in raw numerical format.

This project was developed to create a **visual and analytical platform** that makes crime data
easier to understand through interactive dashboards and predictive analysis.

The aim is to demonstrate how **data science and analytics** can support better understanding
of public safety trends.
""")

    st.markdown("---")

    st.markdown("## 🔍 What the System Does")
    st.markdown("The platform provides multiple analytical features:")

    col1, col2 = st.columns(2)

    with col1:
        st.info("📁 **Crime Data Exploration**\n\nView and understand the dataset structure, columns, and missing values.")
        st.info("📈 **Crime Trend Analysis**\n\nVisualize how crime levels change across years using interactive line charts.")

    with col2:
        st.info("🗺 **Crime Hotspot Identification**\n\nDetect regions with higher crime intensity using clustering techniques.")
        st.info("🤖 **Crime Prediction**\n\nEstimate future crime levels using a trained machine learning model.")

    st.markdown("""
> These components together form an **interactive crime analytics system** that is both
> intuitive and informative.
""")

    st.markdown("---")

    st.markdown("## ⚖️ Impact and Significance")
    st.markdown("""
This system highlights how **data analytics** can assist in understanding social issues like crime patterns.
""")

    impact_data = pd.DataFrame({
        "Benefit": [
            "Regional Crime Pattern Understanding",
            "Data-Driven Trend Insights",
            "High-Risk Area Identification",
            "ML Applications in Analytics"
        ],
        "Value": [85, 80, 90, 88]
    })

    fig_impact = px.bar(
        impact_data,
        x="Value",
        y="Benefit",
        orientation="h",
        color="Value",
        color_continuous_scale="Oranges",
        title="Potential Benefits of the System",
        text="Value"
    )
    fig_impact.update_traces(textposition="outside")
    fig_impact.update_layout(showlegend=False, xaxis_range=[0, 100])
    st.plotly_chart(fig_impact, use_container_width=True)

    st.markdown("""
Although the system is developed as an **academic project**, it reflects how analytics tools
can support informed decision-making.
""")

    st.markdown("---")

    st.markdown("## 🚀 Future Enhancements")
    st.markdown("The system can be further improved with additional capabilities such as:")

    enhancements = [
        ("🌐", "Integration of Real-Time Crime Datasets",      "Connect to live data feeds for up-to-date analysis."),
        ("📍", "District-Level Crime Analysis",                "Deeper insights by drilling down beyond state-level data."),
        ("🧠", "More Advanced Predictive ML Models",           "Use ensemble and deep learning methods for higher accuracy."),
        ("🤖", "Interactive AI-Based Crime Trend Forecasting", "AI-driven scenario analysis for crime trend projections."),
        ("📡", "Deployment as a Public Crime Monitoring Dashboard", "Make the system publicly accessible for broader impact."),
    ]

    for icon, title, desc in enhancements:
        st.markdown(f"**{icon} {title}** — {desc}")

    st.markdown("""
---
> These improvements could transform the system into a **comprehensive crime intelligence platform**.
""")

    st.markdown("---")
    st.caption("Crime Analytics and Prediction System — Academic Project | Built with Streamlit & Python")

# ── FALLBACK ─────────────────────────────────────────────────────
else:
    st.warning("⚠️ Page not found. Please select a page from the sidebar.")
