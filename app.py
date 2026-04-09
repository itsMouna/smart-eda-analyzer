import streamlit as st
import pandas as pd
import plotly.express as px
import io
import json
import requests
from datetime import datetime
import google.generativeai as genai

# ─── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="EDA Analyzer",
    page_icon="📊",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
    <style>
        .main { padding: 2rem; }
        .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
        div[data-testid="metric-container"] {
            background: #1A1A2E;
            border: 1px solid #6C63FF;
            border-radius: 10px;
            padding: 1rem;
        }
        .insight-box {
            background: #1A1A2E;
            border-left: 4px solid #6C63FF;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }
        .chat-user {
            background: #2d2d4e;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            text-align: right;
        }
        .chat-ai {
            background: #1A1A2E;
            border-left: 4px solid #6C63FF;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("EDA Analyzer")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("1. Upload a CSV or Excel file\n2. Clean your data\n3. Explore & visualize\n4. Export report")
    st.markdown("---")
    color = st.selectbox("Chart Color Theme", ["Purple", "Blue", "Green", "Red", "Orange"])
    color_map = {
        "Purple": "#6C63FF",
        "Blue": "#00B4D8",
        "Green": "#2DC653",
        "Red": "#FF4B4B",
        "Orange": "#FF914D"
    }
    chart_color = color_map[color]
    st.markdown("---")

    if st.session_state.df is not None:
        st.markdown("### 📁 Dataset Info")
        st.markdown(f"**Rows:** {st.session_state.df.shape[0]:,}")
        st.markdown(f"**Columns:** {st.session_state.df.shape[1]}")
        st.markdown(f"**Missing:** {st.session_state.df.isnull().sum().sum()}")
        if st.button("🔄 Reset to Original"):
            st.session_state.df = st.session_state.original_df.copy()
            st.rerun()

    st.markdown("---")
    st.caption("Built with Streamlit + Plotly")

# ─── Header ───────────────────────────────────────────────────────
st.title("📊 EDA Analyzer")
st.markdown("Upload any CSV or Excel file and instantly explore, clean, and understand your data.")
st.divider()

# ─── File Upload ──────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
    help="Supports CSV and Excel files up to 200MB"
)

# ─── Load Data ────────────────────────────────────────────────────
def load_data(file):
    if file.name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1")
    else:
        return pd.read_excel(file)

# ─── Auto Insights ────────────────────────────────────────────────
def generate_insights(df):
    insights = []
    numeric_df = df.select_dtypes(include="number")

    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    if missing_pct > 0:
        insights.append(f"⚠️ **{missing_pct:.1f}%** of all values are missing across the dataset.")
    else:
        insights.append("✅ No missing values found — your dataset is complete.")

    dupes = df.duplicated().sum()
    if dupes > 0:
        insights.append(f"⚠️ Found **{dupes:,}** duplicate rows ({dupes/len(df)*100:.1f}% of data).")

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr().abs()
        corr_pairs = corr.unstack()
        corr_pairs = corr_pairs[corr_pairs < 1].sort_values(ascending=False)
        if not corr_pairs.empty:
            top_pair = corr_pairs.index[0]
            top_val = corr_pairs.iloc[0]
            insights.append(f"🔗 Strongest correlation: **{top_pair[0]}** & **{top_pair[1]}** ({top_val:.2f})")

    outlier_cols = []
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)).sum()
        if outliers > 0:
            outlier_cols.append(f"**{col}** ({outliers:,} outliers)")
    if outlier_cols:
        insights.append(f"📌 Outliers detected in: {', '.join(outlier_cols)}")

    skewed = []
    for col in numeric_df.columns:
        skewness = numeric_df[col].skew()
        if abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"
            skewed.append(f"**{col}** ({direction}-skewed)")
    if skewed:
        insights.append(f"📊 Highly skewed columns: {', '.join(skewed)}")

    return insights

# ─── AI Insights ──────────────────────────────────────────────────
def get_ai_insights(df):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash")

        summary = f"""Dataset summary:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns and types: {df.dtypes.astype(str).to_dict()}
- Missing values: {df.isnull().sum().to_dict()}
- Statistics: {df.describe().round(2).to_dict()}"""

        prompt = f"""You are a data analyst. Here is a summary of a dataset:
{summary}

Give 5 concise, specific, and useful insights about this dataset.
Format each as a bullet point starting with an emoji.
Focus on patterns, anomalies, data quality, and what to investigate further."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
# ─── AI Chat ──────────────────────────────────────────────────────
def chat_with_data(df, user_question, history):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash")

        summary = f"""Dataset summary:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {df.dtypes.astype(str).to_dict()}
- Missing values: {df.isnull().sum().to_dict()}
- Statistics: {df.describe().round(2).to_dict()}
- Sample rows: {df.head(3).to_dict()}"""

        history_text = ""
        for h in history:
            history_text += f"User: {h['user']}\nAssistant: {h['ai']}\n\n"

        full_prompt = f"""You are a data analyst assistant. Here is the dataset context:
{summary}

Previous conversation:
{history_text}

User question: {user_question}

Answer clearly and concisely."""

        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
# ─── HTML Report Generator ────────────────────────────────────────
def generate_html_report(df, filename):
    numeric_df = df.select_dtypes(include="number")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    insights = generate_insights(df)

    rows = ""
    for col in df.columns:
        dtype = str(df[col].dtype)
        miss = missing[col]
        miss_p = missing_pct[col]
        unique = df[col].nunique()
        rows += f"<tr><td>{col}</td><td>{dtype}</td><td>{miss} ({miss_p}%)</td><td>{unique}</td></tr>"

    insight_html = "".join([f"<li>{i}</li>" for i in insights])
    stats_html = df.describe().round(2).to_html(classes="table")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EDA Report - {filename}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f9f9f9; color: #333; }}
            h1 {{ color: #6C63FF; }}
            h2 {{ color: #444; border-bottom: 2px solid #6C63FF; padding-bottom: 6px; }}
            .meta {{ background: #fff; padding: 16px; border-radius: 8px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
            .meta span {{ margin-right: 32px; font-size: 18px; }}
            .meta b {{ color: #6C63FF; }}
            table {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
            th {{ background: #6C63FF; color: white; padding: 10px 14px; text-align: left; }}
            td {{ padding: 8px 14px; border-bottom: 1px solid #eee; }}
            tr:hover {{ background: #f0efff; }}
            ul {{ background: #fff; padding: 20px 20px 20px 36px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
            li {{ margin-bottom: 10px; font-size: 15px; }}
            .footer {{ margin-top: 40px; color: #aaa; font-size: 13px; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>📊 EDA Report</h1>
        <div class="meta">
            <span>📁 <b>{filename}</b></span>
            <span>🗓️ {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
            <span>📐 <b>{df.shape[0]:,}</b> rows × <b>{df.shape[1]}</b> columns</span>
            <span>⚠️ <b>{df.isnull().sum().sum()}</b> missing values</span>
            <span>🔁 <b>{df.duplicated().sum()}</b> duplicates</span>
        </div>

        <h2>💡 Key Insights</h2>
        <ul>{insight_html}</ul>

        <h2>🗂️ Column Overview</h2>
        <table>
            <tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique Values</th></tr>
            {rows}
        </table>

        <h2>📈 Statistical Summary</h2>
        {stats_html}

        <div class="footer">Generated by EDA Analyzer · Built with Streamlit & Plotly</div>
    </body>
    </html>
    """
    return html

# ─── Load uploaded file into session ─────────────────────────────
if uploaded_file is not None:
    df_loaded = load_data(uploaded_file)
    if st.session_state.original_df is None or uploaded_file.name not in st.session_state.get("last_file", ""):
        st.session_state.df = df_loaded.copy()
        st.session_state.original_df = df_loaded.copy()
        st.session_state["last_file"] = uploaded_file.name
        st.session_state.chat_history = []

# ─── Main App ─────────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    st.success(f"✅ **{st.session_state.get('last_file', 'file')}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔍 Overview",
        "🧹 Clean Data",
        "📈 Distributions",
        "🔗 Correlations",
        "⚠️ Missing Values",
        "🚨 Outliers",
        "🤖 AI Insights",
        "🔎 Raw Data"
    ])

    # ── TAB 1: Overview ───────────────────────────────────────────
    with tab1:
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplicate Rows", df.duplicated().sum())

        st.divider()
        st.subheader("💡 Auto Insights")
        for insight in generate_insights(df):
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Column Types**")
            type_df = df.dtypes.reset_index()
            type_df.columns = ["Column", "Type"]
            type_df["Type"] = type_df["Type"].astype(str)
            st.dataframe(type_df, use_container_width=True)
        with col_right:
            st.markdown("**Statistical Summary**")
            st.dataframe(df.describe().round(2), use_container_width=True)

        st.divider()
        st.markdown("**📥 Export Report**")
        html_report = generate_html_report(df, st.session_state.get("last_file", "dataset"))
        st.download_button(
            label="⬇️ Download HTML Report",
            data=html_report,
            file_name="eda_report.html",
            mime="text/html"
        )

    # ── TAB 2: Clean Data ─────────────────────────────────────────
    with tab2:
        st.subheader("🧹 Data Cleaning Tools")
        st.info("All changes apply to your working dataset. Use **Reset to Original** in the sidebar to undo everything.")

        # Drop duplicates
        st.markdown("#### 1. Remove Duplicate Rows")
        dupes = df.duplicated().sum()
        st.markdown(f"Found **{dupes}** duplicate rows.")
        if dupes > 0:
            if st.button("🗑️ Drop Duplicate Rows"):
                st.session_state.df = st.session_state.df.drop_duplicates().reset_index(drop=True)
                st.success(f"✅ Removed {dupes} duplicate rows.")
                st.rerun()

        st.divider()

        # Handle missing values
        st.markdown("#### 2. Handle Missing Values")
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        if not cols_with_missing:
            st.success("✅ No missing values to handle!")
        else:
            selected_col = st.selectbox("Select column to fix", cols_with_missing)
            strategy = st.radio("Fill strategy", ["Mean", "Median", "Mode", "Custom Value", "Drop Rows"], horizontal=True)
            custom_val = ""
            if strategy == "Custom Value":
                custom_val = st.text_input("Enter custom value")

            if st.button("✅ Apply Fix"):
                col_dtype = st.session_state.df[selected_col].dtype
                if strategy == "Mean":
                    st.session_state.df[selected_col].fillna(st.session_state.df[selected_col].mean(), inplace=True)
                elif strategy == "Median":
                    st.session_state.df[selected_col].fillna(st.session_state.df[selected_col].median(), inplace=True)
                elif strategy == "Mode":
                    st.session_state.df[selected_col].fillna(st.session_state.df[selected_col].mode()[0], inplace=True)
                elif strategy == "Custom Value" and custom_val != "":
                    try:
                        val = float(custom_val) if col_dtype in ["float64", "int64"] else custom_val
                    except:
                        val = custom_val
                    st.session_state.df[selected_col].fillna(val, inplace=True)
                elif strategy == "Drop Rows":
                    st.session_state.df.dropna(subset=[selected_col], inplace=True)
                    st.session_state.df.reset_index(drop=True, inplace=True)
                st.success(f"✅ Applied '{strategy}' to **{selected_col}**")
                st.rerun()

        st.divider()

        # Drop columns
        st.markdown("#### 3. Drop Columns")
        cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
        if cols_to_drop:
            if st.button("🗑️ Drop Selected Columns"):
                st.session_state.df.drop(columns=cols_to_drop, inplace=True)
                st.success(f"✅ Dropped: {', '.join(cols_to_drop)}")
                st.rerun()

        st.divider()

        # Rename columns
        st.markdown("#### 4. Rename a Column")
        col_to_rename = st.selectbox("Select column to rename", df.columns.tolist())
        new_name = st.text_input("New column name")
        if new_name and st.button("✏️ Rename Column"):
            st.session_state.df.rename(columns={col_to_rename: new_name}, inplace=True)
            st.success(f"✅ Renamed **{col_to_rename}** → **{new_name}**")
            st.rerun()

        st.divider()

        # Download cleaned data
        st.markdown("#### 5. Download Cleaned Dataset")
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # ── TAB 3: Distributions ──────────────────────────────────────
    with tab3:
        st.subheader("Column Distributions")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_cols:
            st.markdown("#### Numeric Columns")
            selected_num = st.selectbox("Select a numeric column", numeric_cols)
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.histogram(df, x=selected_num, nbins=30,
                                   title=f"Distribution of {selected_num}",
                                   color_discrete_sequence=[chart_color])
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig2 = px.box(df, y=selected_num,
                              title=f"Box Plot of {selected_num}",
                              color_discrete_sequence=[chart_color])
                st.plotly_chart(fig2, use_container_width=True)

            # Time series detection
            date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            possible_date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]
            if possible_date_cols:
                st.divider()
                st.markdown("#### 📅 Time Series")
                date_col = st.selectbox("Select date/time column", possible_date_cols)
                try:
                    df_ts = df.copy()
                    df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                    df_ts = df_ts.sort_values(date_col)
                    fig_ts = px.line(df_ts, x=date_col, y=selected_num,
                                    title=f"{selected_num} over {date_col}",
                                    color_discrete_sequence=[chart_color])
                    fig_ts.update_traces(line=dict(width=2))
                    st.plotly_chart(fig_ts, use_container_width=True)
                except:
                    st.warning("Could not parse this column as a date.")

        st.divider()
        if len(numeric_cols) >= 2:
            st.markdown("#### Scatter Plot Builder")
            col_x, col_y = st.columns(2)
            with col_x:
                x_axis = st.selectbox("X Axis", numeric_cols, index=0)
            with col_y:
                y_axis = st.selectbox("Y Axis", numeric_cols, index=1)
            color_by = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
            fig_scatter = px.scatter(
                df, x=x_axis, y=y_axis,
                color=None if color_by == "None" else color_by,
                title=f"{x_axis} vs {y_axis}",
                trendline=None,
                color_discrete_sequence=[chart_color]
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.divider()
            st.markdown("#### Pair Plot (all numeric columns)")
            if len(numeric_cols) <= 6:
                fig_pair = px.scatter_matrix(df, dimensions=numeric_cols,
                                              color_discrete_sequence=[chart_color])
                fig_pair.update_layout(height=700)
                st.plotly_chart(fig_pair, use_container_width=True)
            else:
                selected_pair_cols = st.multiselect("Select up to 6 columns for pair plot", numeric_cols, default=numeric_cols[:4])
                if len(selected_pair_cols) >= 2:
                    fig_pair = px.scatter_matrix(df, dimensions=selected_pair_cols,
                                                  color_discrete_sequence=[chart_color])
                    fig_pair.update_layout(height=700)
                    st.plotly_chart(fig_pair, use_container_width=True)

        if categorical_cols:
            st.divider()
            st.markdown("#### Categorical Columns")
            selected_cat = st.selectbox("Select a categorical column", categorical_cols)
            value_counts = df[selected_cat].value_counts().reset_index()
            value_counts.columns = [selected_cat, "Count"]
            col_bar, col_pie = st.columns(2)
            with col_bar:
                fig3 = px.bar(value_counts.head(20), x=selected_cat, y="Count",
                              title=f"Top Values in {selected_cat}",
                              color_discrete_sequence=[chart_color])
                st.plotly_chart(fig3, use_container_width=True)
            with col_pie:
                fig_pie = px.pie(value_counts.head(10), names=selected_cat, values="Count",
                                 title=f"Share of {selected_cat}",
                                 color_discrete_sequence=px.colors.sequential.Purples_r)
                st.plotly_chart(fig_pie, use_container_width=True)

    # ── TAB 4: Correlations ───────────────────────────────────────
    with tab4:
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numeric columns to show correlations.")
        else:
            corr = numeric_df.corr().round(2)
            fig = px.imshow(corr, text_auto=True,
                            color_continuous_scale="Viridis",
                            title="Correlation Matrix")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("#### Top Correlated Pairs")
            corr_pairs = corr.unstack().reset_index()
            corr_pairs.columns = ["Column A", "Column B", "Correlation"]
            corr_pairs = corr_pairs[corr_pairs["Column A"] < corr_pairs["Column B"]]
            corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()
            corr_pairs = corr_pairs.sort_values("Abs Correlation", ascending=False).drop(columns="Abs Correlation")
            corr_pairs["Correlation"] = corr_pairs["Correlation"].round(3)
            st.dataframe(corr_pairs.head(10), use_container_width=True)

    # ── TAB 5: Missing Values ─────────────────────────────────────
    with tab5:
        st.subheader("Missing Values Analysis")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        missing = missing[missing["Missing Count"] > 0].sort_values("Missing %", ascending=False)

        if missing.empty:
            st.success("🎉 No missing values found in your dataset!")
        else:
            st.dataframe(missing, use_container_width=True)
            fig = px.bar(missing, x="Column", y="Missing %",
                         title="Missing Values by Column (%)",
                         color="Missing %",
                         color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 6: Outliers ───────────────────────────────────────────
    with tab6:
        st.subheader("Outlier Detection (IQR Method)")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns found.")
        else:
            selected_out = st.selectbox("Select column", numeric_cols)
            col_data = df[selected_out].dropna()
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[selected_out] < lower) | (df[selected_out] > upper)]

            col1, col2, col3 = st.columns(3)
            col1.metric("Lower Bound", f"{lower:.2f}")
            col2.metric("Upper Bound", f"{upper:.2f}")
            col3.metric("Outliers Found", len(outliers))

            fig = px.box(df, y=selected_out,
                         title=f"Outliers in {selected_out}",
                         color_discrete_sequence=[chart_color])
            st.plotly_chart(fig, use_container_width=True)

            if not outliers.empty:
                st.markdown("**Outlier Rows:**")
                st.dataframe(outliers, use_container_width=True)

                if st.button("🗑️ Remove Outliers from Dataset"):
                    before = len(st.session_state.df)
                    Q1_ = st.session_state.df[selected_out].quantile(0.25)
                    Q3_ = st.session_state.df[selected_out].quantile(0.75)
                    IQR_ = Q3_ - Q1_
                    st.session_state.df = st.session_state.df[
                        (st.session_state.df[selected_out] >= Q1_ - 1.5 * IQR_) &
                        (st.session_state.df[selected_out] <= Q3_ + 1.5 * IQR_)
                    ].reset_index(drop=True)
                    after = len(st.session_state.df)
                    st.success(f"✅ Removed {before - after} outlier rows from **{selected_out}**")
                    st.rerun()

    # ── TAB 7: AI Insights ────────────────────────────────────────
    with tab7:
        st.subheader("🤖 AI-Powered Analysis")

        ai_tab1, ai_tab2 = st.tabs(["✨ Auto Insights", "💬 Chat with your Data"])

        with ai_tab1:
            st.markdown("Get gemini's expert analysis of your dataset.")
            if st.button("✨ Generate AI Insights", type="primary"):
                with st.spinner("gemini is analyzing your data..."):
                    ai_response = get_ai_insights(df)
                    st.markdown(ai_response)

        with ai_tab2:
            st.markdown("Ask anything about your dataset and gemini will answer.")

            for msg in st.session_state.chat_history:
                st.markdown(f'<div class="chat-user">🧑 {msg["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-ai">🤖 {msg["ai"]}</div>', unsafe_allow_html=True)

            user_q = st.text_input("Ask a question about your data...", key="chat_input")
            if st.button("Send 💬") and user_q.strip():
                with st.spinner("Thinking..."):
                    ai_answer = chat_with_data(df, user_q, st.session_state.chat_history)
                    st.session_state.chat_history.append({"user": user_q, "ai": ai_answer})
                    st.rerun()

            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()

    # ── TAB 8: Raw Data ───────────────────────────────────────────
    with tab8:
        st.subheader("Raw Data")
        search = st.text_input("🔍 Filter rows (search any value)")
        display_df = df
        if search:
            mask = df.astype(str).apply(lambda row: row.str.contains(search, case=False)).any(axis=1)
            display_df = df[mask]
            st.markdown(f"Showing **{len(display_df)}** matching rows")

        st.dataframe(display_df, use_container_width=True)
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv,
            file_name="data_export.csv",
            mime="text/csv"
        )

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👆 Upload a CSV or Excel file to get started.")
        st.markdown("**Supported formats:** CSV, Excel (.xlsx)")
        st.markdown("**Max file size:** 200MB")