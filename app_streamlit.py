import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deliveroo Campaign Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍔"
)

# --- BRAND COLORS (Deliveroo Palette) ---
DELIVEROO_TEAL = "#00CCBC"  # Robin's Egg Blue
DELIVEROO_DARK = "#2E3333"  # Deep Outer Space
DELIVEROO_GREY = "#F5F5F5"  # Light background

# --- CSS FOR STYLING ---
st.markdown(f"""
<style>
    .metric-card {{
        background-color: {DELIVEROO_GREY};
        border-left: 5px solid {DELIVEROO_TEAL};
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }}
    h1, h2, h3 {{
        color: {DELIVEROO_DARK};
    }}
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING ---
st.sidebar.image("deliveroo.png", width=150)
st.sidebar.header("📂 Data Setup")
uploaded_file = st.sidebar.file_uploader("Upload 'ClickPrediction' CSV", type=["csv"])

if uploaded_file is None:
    st.info("👋 Welcome to the Deliveroo Ad Optimizer! Please upload your CSV file in the sidebar.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Logic for Tier
    def get_tier(prob):
        if prob >= 0.9: return "High Priority (Hot)"
        elif prob >= 0.5: return "Medium Priority (Warm)"
        else: return "Low Priority (Cold)"

    df['Lead_Tier'] = df['Predicted_Click_Probability'].apply(get_tier)

    # Logic for Heatmap Bins
    df['Day_Part'] = pd.cut(df['Daytime'], bins=[0, 0.33, 0.66, 1.0], labels=["Morning", "Afternoon", "Evening"])

    # Weekday Ordering
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if 'Weekday' in df.columns:
        df['Weekday'] = pd.Categorical(df['Weekday'], categories=days, ordered=True)

    return df

df = load_data(uploaded_file)

# --- 2. SIDEBAR FILTERS ---
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Simulation Parameters")

# Threshold Slider
threshold = st.sidebar.slider("Min. Probability Threshold", 0.0, 1.0, 0.5, 0.05)

# Region Filter
all_regions = ["All"] + sorted(list(df['Region'].unique())) if 'Region' in df.columns else ["All"]
selected_region = st.sidebar.selectbox("Filter by Region", all_regions)

# Applying Filters
df_region = df if selected_region == "All" else df[df['Region'] == selected_region]
df_target = df_region[df_region['Predicted_Click_Probability'] >= threshold]

# --- DASHBOARD HEADER ---
st.title("🍔 Deliveroo Campaign Optimizer")
st.markdown(f"**Targeting Scenario:** Users in *{selected_region}* with > *{int(threshold*100)}%* probability.")

# ==============================================================================
# BLOCK A: EXECUTIVE SUMMARY
# ==============================================================================
st.header("1. Executive Summary")

# --- 0. OFFICIAL BRAND COLORS ---
DELIVEROO_TEAL = "#00CCBC"  # Robin's Egg Blue
DELIVEROO_DARK = "#2E3333"  # Outer Space

# --- 1. METRICS CALCULATION ---
total_leads = len(df_region)
targeted = len(df_target)
proj_conv = df_target['Predicted_Click_Probability'].sum()

# Efficiency Score
total_potential_conv = df_region['Predicted_Click_Probability'].sum()
eff_score = (proj_conv / total_potential_conv * 100) if total_potential_conv > 0 else 0

# Volume Score
vol_score = (targeted / total_leads * 100) if total_leads > 0 else 0

# Avg Propensity
avg_propensity = df_target['Predicted_Click_Probability'].mean() if targeted > 0 else 0

# --- 2. KPI CARDS ---
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Users (Pool)",
    f"{total_leads:,}",
    help="Total number of leads available in the selected region before filtering."
)

c2.metric(
    "Targeted Users",
    f"{targeted:,}",
    f"{vol_score:.1f}% of Pool",
    help="Number of users selected by your targeting rule."
)

c3.metric(
    "Projected Conversions",
    f"{int(proj_conv):,}",
    help="Expected number of conversions (sum of probabilities) from the targeted users."
)

c4.metric(
    "Avg. Propensity",
    f"{avg_propensity:.1%}",
    help="The average click probability among the users you have selected."
)

# --- 3. CHARTS ROW ---
c_chart1, c_chart2 = st.columns([1, 2])

with c_chart1:
    # Donut Chart with Tooltip Title
    st.markdown("##### Conversion Mix (Targeted)",
                help="Distribution of targeted users by predicted likelihood of conversion.")

    counts = df_target['Predicted_Click'].value_counts().reset_index()
    if not counts.empty:
        counts.columns = ['Result', 'Count']
        counts['Result'] = counts['Result'].replace({1: "Likely Conversion", 0: "No Conversion"})

        # Title is removed from px.pie to avoid duplication
        fig_donut = px.pie(counts, values='Count', names='Result', hole=0.6,
                           color='Result',
                           color_discrete_map={"Likely Conversion": DELIVEROO_TEAL, "No Conversion": DELIVEROO_DARK})
        st.plotly_chart(fig_donut, use_container_width=True)

with c_chart2:
    # Simulation Bar Chart
    st.markdown("##### 💡 Simulation Insight: Volume vs. Efficiency")
    st.caption("Visualizes the trade-off between the **Budget** you spend (Dark) and the **Value** you capture (Teal). "
               "A large gap between the bars indicates a highly efficient strategy.")

    sim_data = pd.DataFrame({
        'Metric': ['Budget Spent (Volume)', 'Conversions Captured (Value)'],
        'Percentage': [vol_score, eff_score],
        'Type': ['Cost', 'Value']
    })

    fig_sim = px.bar(sim_data, x='Percentage', y='Metric', orientation='h',
                     color='Type', text='Percentage',
                     color_discrete_map={'Cost': DELIVEROO_DARK, 'Value': DELIVEROO_TEAL})

    fig_sim.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_sim.update_layout(xaxis_range=[0, 115], height=250, margin=dict(t=0)) # Adjusted margin for alignment
    st.plotly_chart(fig_sim, use_container_width=True)

st.markdown("---")

# ==============================================================================
# BLOCK B: STRATEGIC INSIGHTS
# ==============================================================================
st.header("2. Strategic Insights")
st.markdown("Validating model drivers to understand audience behavior.")

# --- ROW 1: CHANNELS & CONTENT (The "Where" and "What") ---
col_b1, col_b2 = st.columns(2)

with col_b1:
    st.markdown("##### Social Network Effectiveness", help="Which platforms drive higher quality leads?")
    # Group by Social Network
    social_stats = df_region.groupby("Social_Network")["Predicted_Click_Probability"].mean().reset_index()

    fig_social = px.bar(social_stats, x="Social_Network", y="Predicted_Click_Probability",
                        color="Social_Network",
                        text_auto='.1%',
                        color_discrete_sequence=[DELIVEROO_TEAL, DELIVEROO_DARK, "#888888"])

    fig_social.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Avg. Probability", height=300)
    fig_social.update_yaxes(range=[0, 1.1]) # Fix range to make comparisons fair
    st.plotly_chart(fig_social, use_container_width=True)
    st.caption(" Insight: Prioritize spend on channels with higher bars.")

with col_b2:
    st.markdown("##### Restaurant Type Performance", help="Which cuisines drive the highest user intent?")

    # Handle missing values for visualization
    df_chart = df_region.copy()
    df_chart['Restaurant_Type'] = df_chart['Restaurant_Type'].fillna("Unknown")

    # Group by Restaurant Type
    rest_stats = df_chart.groupby("Restaurant_Type")["Predicted_Click_Probability"].mean().reset_index()
    rest_stats = rest_stats.sort_values(by="Predicted_Click_Probability", ascending=False)

    fig_rest = px.bar(rest_stats, x="Predicted_Click_Probability", y="Restaurant_Type",
                      orientation='h',
                      text_auto='.1%',
                      color="Predicted_Click_Probability",
                      color_continuous_scale=[(0, DELIVEROO_DARK), (1, DELIVEROO_TEAL)])

    fig_rest.update_layout(showlegend=False, xaxis_title="Avg. Probability", yaxis_title=None, height=300)
    fig_rest.update_xaxes(range=[0, 1.1])
    st.plotly_chart(fig_rest, use_container_width=True)
    st.caption(" Insight: Feature high-performing cuisines in your ad creatives.")

# --- ROW 2: TIMING & TECHNICAL (The "When" and "How") ---
col_b3, col_b4 = st.columns([2, 1])

with col_b3:
    st.markdown("##### Timing Heatmap", help="Darker areas indicate lower intent, while bright Teal areas indicate high-intent times to bid.")
    if 'Day_Part' in df_region.columns:
        heatmap_data = df_region.pivot_table(index="Weekday", columns="Day_Part",
                                             values="Predicted_Click_Probability", aggfunc="mean")

        # Custom color scale: Dark -> Teal gradient
        fig_heat = px.imshow(heatmap_data, text_auto=".0%", aspect="auto",
                             color_continuous_scale=[(0, DELIVEROO_DARK), (1, DELIVEROO_TEAL)])

        fig_heat.update_layout(height=300)
        st.plotly_chart(fig_heat, use_container_width=True)

with col_b4:
    st.markdown("##### Carrier Influence", help="Performance by mobile network provider.")

    # 1. Group Data
    carrier_stats = df_region.groupby("Carrier")["Predicted_Click_Probability"].mean().reset_index()
    carrier_stats = carrier_stats.sort_values(by="Predicted_Click_Probability", ascending=False)

    # 2. Define Custom Colors (Map specific carriers to specific colors if desired, or use a list)
    # This list cycles through: Teal -> Dark -> Dark -> Gray (Free)
    # You can also map explicitly if you know the order, but a list is safer for dynamic data
    custom_palette = [DELIVEROO_TEAL, DELIVEROO_DARK, "#36454F", "#D3D3D3"]

    # 3. Create Chart
    fig_carrier = px.bar(carrier_stats, x="Carrier", y="Predicted_Click_Probability",
                         color="Carrier",
                         text_auto='.1%', # Adds the percentage number on the bar
                         color_discrete_sequence=custom_palette)

    # 4. Clean Layout
    fig_carrier.update_layout(showlegend=False,
                              margin=dict(t=20, b=20),
                              height=300,
                              yaxis_title=None,
                              xaxis_title=None)

    fig_carrier.update_yaxes(showticklabels=False) # Hide y-axis numbers since we have labels
    st.plotly_chart(fig_carrier, use_container_width=True)

# ==============================================================================
# BLOCK C: OPERATIONAL TOOL
# ==============================================================================
st.header("3. Operational Tool (Export List)")
st.markdown("Download the specific list of targeted users.")

# Added 'Predicted_Click' and re-ordered columns
show_cols = ['Predicted_Click', 'Predicted_Click_Probability', 'Lead_Tier',
             'Region', 'Social_Network', 'Weekday', 'Carrier']

# Show table
st.dataframe(df_target[show_cols].sort_values(by="Predicted_Click_Probability", ascending=False), use_container_width=True)

# Export CSV
csv = df_target.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Targeted Lead List (CSV)",
    data=csv,
    file_name=f"Deliveroo_Target_Leads_{selected_region}.csv",
    mime="text/csv",
)

st.header("4. Shapley explanation of each prediction")

with st.expander("📦 Shapley values info (upload)", expanded=True):
    shap_file = st.file_uploader(
        "Upload the SHAP-enriched CSV (Prediction / Predict_Proba / Top_Shapley_1..10 / Value_Shapley_1..10)",
        type=["csv"],
        key="shap_upload"
    )

    if shap_file is None:
        st.info("Upload your SHAP-enriched CSV to visualize explanations.")
        st.stop()

    shap_df = pd.read_csv(shap_file)

    # ✅ Drop 'Unnamed: 0' / any unnamed index column
    shap_df = shap_df.loc[:, ~shap_df.columns.str.contains(r"^Unnamed")]

    st.success("✅ SHAP file loaded.")
    st.dataframe(shap_df.head(30), use_container_width=True)

    # Optional re-download
    csv_bytes = shap_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download cleaned SHAP CSV",
        data=csv_bytes,
        file_name="SHAP_enriched_predictions_clean.csv",
        mime="text/csv"
    )

st.markdown("---")
st.subheader("🔎 Inspect an individual prediction")

# ✅ Manual index input (0..1999) + safe bounds to your real df size
max_idx = min(1999, len(shap_df) - 1)
row_id = st.number_input(
    f"Choose a row index (0 to {max_idx})",
    min_value=0,
    max_value=max_idx,
    value=0,
    step=1
)

row = shap_df.iloc[int(row_id)]

# KPIs
c1, c2 = st.columns(2)
c1.metric("Prediction", int(row["Prediction"]) if "Prediction" in row else "N/A")
c2.metric("Predict_Proba", f"{float(row['Predict_Proba']):.4f}" if "Predict_Proba" in row else "N/A")

# Required columns check
top_k = 10
feat_cols = [f"Top_Shapley_{k}" for k in range(1, top_k + 1)]
val_cols  = [f"Value_Shapley_{k}" for k in range(1, top_k + 1)]

missing = [c for c in (feat_cols + val_cols) if c not in shap_df.columns]
if missing:
    st.error("Missing required columns:\n" + "\n".join(missing))
    st.stop()

# Build user SHAP table
features = [row[c] for c in feat_cols]
values = np.array([row[c] for c in val_cols], dtype=float)

user_shap_df = pd.DataFrame({
    "Feature": features,
    "SHAP value": values,
    "Abs impact": np.abs(values),
    "Direction": np.where(values >= 0, "Pushes toward class 1", "Pushes toward class 0")
}).sort_values("Abs impact", ascending=True)  # ascending looks nicer for h-bar

# Plot
fig = px.bar(
    user_shap_df,
    x="SHAP value",
    y="Feature",
    orientation="h",
    color="Abs impact",
    color_continuous_scale=[(0, "#2E3333"), (1, "#00CCBC")],  # dark -> teal
    hover_data=["Abs impact", "Direction"],
    title=f"Top 10 SHAP drivers — row {int(row_id)}"
)
fig.update_layout(height=450, margin=dict(t=50, b=20), coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# Table
st.markdown("##### Top SHAP drivers (table)")
st.dataframe(
    user_shap_df.sort_values("Abs impact", ascending=False)[["Feature", "SHAP value", "Direction"]],
    use_container_width=True
)
