import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="DV ScreenSense", layout="wide")
st.title("📱 DV ScreenSense — Indian Kids' Screen Time Insights")
st.caption("Data cleaning, EDA & dashboarding on the Indian Kids Screentime 2025 dataset")

# ---------------------------------------------------------------
# DATA LOADING & CLEANING
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df["Health_Impacts"] = df["Health_Impacts"].fillna("None")
    df["AgeGroup"] = pd.cut(df["Age"], bins=[6, 12, 16, 20],
                             labels=["7-12", "13-16", "17-20"])
    return df

df = load_data()

# ---------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------
st.sidebar.header("Filters")
gender_f = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
loc_f = st.sidebar.multiselect("Location", df["Urban_or_Rural"].unique(), default=list(df["Urban_or_Rural"].unique()))
device_f = st.sidebar.multiselect("Primary Device", df["Primary_Device"].unique(), default=list(df["Primary_Device"].unique()))

f = df[df["Gender"].isin(gender_f) & df["Urban_or_Rural"].isin(loc_f) & df["Primary_Device"].isin(device_f)]

# ---------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Children Surveyed", f"{len(f):,}")
c2.metric("Avg Screen Time", f"{f['Avg_Daily_Screen_Time_hr'].mean():.2f} hr")
c3.metric("% Exceeding Limit", f"{f['Exceeded_Recommended_Limit'].mean()*100:.1f}%")
c4.metric("Avg Educational Ratio", f"{f['Educational_to_Recreational_Ratio'].mean():.2f}")

st.divider()

# ---------------------------------------------------------------
# ROW 1: Screen time by gender, device mix
# ---------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Average Screen Time by Gender")
    g = f.groupby("Gender", observed=True)["Avg_Daily_Screen_Time_hr"].mean().reset_index()
    fig = px.bar(g, x="Gender", y="Avg_Daily_Screen_Time_hr", color="Gender", text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Device Mix by Location")
    dev = f.groupby(["Urban_or_Rural", "Primary_Device"]).size().reset_index(name="Count")
    fig = px.bar(dev, x="Urban_or_Rural", y="Count", color="Primary_Device", barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# ROW 2: Distribution — boxplot & histogram
# ---------------------------------------------------------------
col3, col4 = st.columns(2)
with col3:
    st.subheader("Screen Time Distribution by Age Group & Gender")
    fig = px.box(f, x="AgeGroup", y="Avg_Daily_Screen_Time_hr", color="Gender")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Overall Screen Time Distribution")
    fig = px.histogram(f, x="Avg_Daily_Screen_Time_hr", nbins=30, marginal="box")
    fig.add_vline(x=2, line_dash="dash", line_color="red",
                   annotation_text="Typical recommended limit (~2hr)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# ROW 3: Corrected Health Impacts (fixed multi-label bug)
# ---------------------------------------------------------------
st.subheader("Health Impacts — Corrected (individual impacts, not combo strings)")
st.caption("Note: original combo strings like 'Poor Sleep, Eye Strain' were split into individual impacts so counts aren't undercounted.")
impacts = f["Health_Impacts"].str.split(",").explode().str.strip()
impact_counts = impacts.value_counts().reset_index()
impact_counts.columns = ["Impact", "Count"]
fig = px.bar(impact_counts, x="Count", y="Impact", orientation="h", color="Count", color_continuous_scale="viridis")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# ROW 4: Exceeded limit breakdown
# ---------------------------------------------------------------
st.subheader("% Exceeding Recommended Screen Time Limit")
col5, col6, col7 = st.columns(3)
for col, groupcol, label in zip([col5, col6, col7], ["AgeGroup", "Gender", "Urban_or_Rural"],
                                  ["by Age Group", "by Gender", "by Location"]):
    with col:
        rate = f.groupby(groupcol, observed=True)["Exceeded_Recommended_Limit"].mean().reset_index()
        rate["Exceeded_Recommended_Limit"] *= 100
        fig = px.bar(rate, x=groupcol, y="Exceeded_Recommended_Limit", text_auto=".1f",
                     title=label, labels={"Exceeded_Recommended_Limit": "% exceeded"})
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# ROW 5: Educational ratio insight
# ---------------------------------------------------------------
st.subheader("Does Educational Content Offset Screen Time or Health Impact?")
col8, col9 = st.columns(2)
with col8:
    fig = px.scatter(f, x="Educational_to_Recreational_Ratio", y="Avg_Daily_Screen_Time_hr",
                      color="Exceeded_Recommended_Limit", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)
    corr = f["Educational_to_Recreational_Ratio"].corr(f["Avg_Daily_Screen_Time_hr"])
    st.metric("Correlation coefficient", f"{corr:.3f}")

with col9:
    has_impact = f["Health_Impacts"] != "None"
    tmp = f.copy()
    tmp["Has_Impact"] = has_impact.map({True: "Has Health Impact", False: "No Impact"})
    fig = px.box(tmp, x="Has_Impact", y="Educational_to_Recreational_Ratio", color="Has_Impact")
    st.plotly_chart(fig, use_container_width=True)

st.info(
    "**Finding:** Educational content ratio shows almost no correlation with either total "
    "screen time or the presence of a health impact in this dataset — more educational "
    "content does not appear to offset the risks of high screen time. This is a genuine "
    "null result, not a data gap."
)

# ---------------------------------------------------------------
# SQL PERFORMANCE LAB
# ---------------------------------------------------------------
import sqlite3
import time

st.divider()
st.header("🔍 SQL Performance Lab")
st.caption(
    "Demonstrates query optimization and indexing tradeoffs. The dataset is "
    "synthetically scaled 60x (~580K rows) purely so timing differences are "
    "measurable — this section is a performance benchmark, not a data finding."
)

@st.cache_resource
def build_sql_demo():
    scaled = pd.concat([df] * 60, ignore_index=True)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    scaled.to_sql("screentime", conn, if_exists="replace", index=False)
    return conn

conn = build_sql_demo()
cur = conn.cursor()

def timed_query(sql, n=5):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        cur.execute(sql).fetchall()
        times.append(time.perf_counter() - t0)
    return min(times) * 1000  # ms

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Case 1: Selective filter — index helps")
    q1 = "SELECT COUNT(*), AVG(Avg_Daily_Screen_Time_hr) FROM screentime WHERE Age = 9"
    st.code(q1, language="sql")
    t1_before = timed_query(q1)
    plan1_before = cur.execute("EXPLAIN QUERY PLAN " + q1).fetchall()
    cur.execute("DROP INDEX IF EXISTS idx_age")
    cur.execute("CREATE INDEX idx_age ON screentime(Age)")
    t1_after = timed_query(q1)
    plan1_after = cur.execute("EXPLAIN QUERY PLAN " + q1).fetchall()
    cur.execute("DROP INDEX idx_age")

    fig1 = px.bar(x=["No index", "With index"], y=[t1_before, t1_after],
                  labels={"x": "", "y": "Time (ms)"}, title=f"{t1_before/t1_after:.1f}x faster with index")
    st.plotly_chart(fig1, use_container_width=True)
    st.write(f"**Before:** `{plan1_before[0][3]}`")
    st.write(f"**After:** `{plan1_after[0][3]}`")
    st.success("Exact-match filter on a specific age (~7% of rows) → SQLite switches from a full SCAN to an indexed SEARCH.")

with col_b:
    st.subheader("Case 2: Low-selectivity filter — index hurts")
    q2 = "SELECT AVG(Avg_Daily_Screen_Time_hr) FROM screentime WHERE Exceeded_Recommended_Limit = 1"
    st.code(q2, language="sql")
    t2_before = timed_query(q2)
    plan2_before = cur.execute("EXPLAIN QUERY PLAN " + q2).fetchall()
    cur.execute("DROP INDEX IF EXISTS idx_exceed")
    cur.execute("CREATE INDEX idx_exceed ON screentime(Exceeded_Recommended_Limit)")
    t2_after = timed_query(q2)
    plan2_after = cur.execute("EXPLAIN QUERY PLAN " + q2).fetchall()
    cur.execute("DROP INDEX idx_exceed")

    fig2 = px.bar(x=["No index", "With index"], y=[t2_before, t2_after],
                  labels={"x": "", "y": "Time (ms)"}, title=f"{t2_before/t2_after:.2f}x with index (worse)")
    st.plotly_chart(fig2, use_container_width=True)
    st.write(f"**Before:** `{plan2_before[0][3]}`")
    st.write(f"**After:** `{plan2_after[0][3]}`")
    st.warning(
        "~85% of rows match this filter, and the query needs a non-indexed column "
        "(Avg_Daily_Screen_Time_hr). The index forces random row lookups for most "
        "of the table instead of one cheap sequential scan — so it's slower, not faster. "
        "Indexing low-selectivity columns for non-covering queries is a classic "
        "'looks like it should help but doesn't' case at scale."
    )

# ---------------------------------------------------------------
# RAW DATA
# ---------------------------------------------------------------
with st.expander("View filtered raw data"):
    st.dataframe(f)

st.caption("Data source: Kaggle — Indian Kids Screentime 2025. Built with Python, Pandas & Plotly.")
