"""
app.py
======
Streamlit web interface for TrialStat AI.

Key fix: uses st.session_state to persist the dataframe across rerenders.
Without this, clicking any widget resets the page and loses the uploaded data.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from decision_engine import select_test, TEST_DISPLAY_NAMES
from stat_tests import run_test
from report_generator import generate_report, PROVIDERS

st.set_page_config(
    page_title="TrialStat AI",
    page_icon="🧬",
    layout="centered",
)

# ── Initialise session state (runs once per session) ─────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-v1-...",
        help="Get a free key at openrouter.ai. All models below are free tier.",
    )

    provider = st.selectbox(
        "Model",
        options=list(PROVIDERS.keys()),
    )

    if api_key:
        st.success(f"✓ Key set — using {provider}")
    else:
        st.warning(
            "No API key. Statistical analysis still runs normally. "
            "Only the AI report step needs a key. "
            "Get a free key at openrouter.ai"
        )

    st.divider()
    st.caption("TrialStat AI · Automated Clinical Trial Statistical Analysis")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🧬 TrialStat AI")
st.caption(
    "Automated Clinical Trial Statistical Analysis · "
    "Automatic test selection · Plain-language report"
)

with st.expander("How does this work?", expanded=False):
    st.markdown("""
    **Statistical decision (rule engine — no AI)**
    Shapiro-Wilk normality test + Levene variance test → decision tree →
    recommends: Independent t-test / Welch's t-test / Mann-Whitney U + Bootstrap.

    **Report generation (LLM — only this step uses AI)**
    Statistical output → OpenRouter-hosted LLM → plain-language report for stakeholders.
    """)

st.divider()

# ── Data loading ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a CSV file (one column for group labels, one for numeric values)",
    type="csv",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    demo_clicked = st.button("No data? Try the goat vaccine demo dataset")
with col_clear:
    if st.button("Clear data"):
        st.session_state.df = None
        st.session_state.demo_loaded = False
        st.rerun()

# Update session state based on input — this persists across rerenders
if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.demo_loaded = False
elif demo_clicked:
    st.session_state.df = pd.read_csv("sample_data/goat_worms.csv")
    st.session_state.demo_loaded = True

df = st.session_state.df

if st.session_state.demo_loaded:
    st.info("Demo data loaded: goat deworming vaccine trial (control vs. vaccinated)")

# ── Analysis ──────────────────────────────────────────────────────────────────
if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        group_col = st.selectbox("Group column", df.columns, index=0)
    with col2:
        value_col = st.selectbox(
            "Value column", df.columns,
            index=min(1, len(df.columns) - 1),
        )

    groups = df[group_col].dropna().unique()

    if len(groups) != 2:
        st.error(
            f"Column '{group_col}' has {len(groups)} unique values. "
            f"TrialStat AI requires exactly two groups."
        )
    else:
        label_a, label_b = groups[0], groups[1]
        group_a = df[df[group_col] == label_a][value_col].dropna().values
        group_b = df[df[group_col] == label_b][value_col].dropna().values

        context = st.text_area(
            "Business context (optional — personalises the AI report)",
            placeholder=(
                "e.g. A pharmaceutical company wants to know whether a new "
                "deworming vaccine reduces parasite counts in goats..."
            ),
        )

        if st.button("Run analysis", type="primary"):

            # Step 1: assumption checks
            with st.spinner("Checking statistical assumptions..."):
                test_name, diagnostics = select_test(group_a, group_b)

            st.subheader("① Assumption diagnostics")
            norm = diagnostics["normality"]
            st.write(f"**Normality (Shapiro-Wilk):** {norm['interpretation']}")
            if "group_a_p_value" in norm:
                st.write(
                    f"　{label_a}: p = {norm['group_a_p_value']}　"
                    f"{label_b}: p = {norm['group_b_p_value']}"
                )
            if "variance_homogeneity" in diagnostics:
                var = diagnostics["variance_homogeneity"]
                st.write(
                    f"**Variance homogeneity (Levene's):** "
                    f"{var['interpretation']} (p = {var['p_value']})"
                )
            if diagnostics.get("is_small_sample"):
                st.warning(
                    f"⚠️ Small sample (minimum group size: "
                    f"{min(diagnostics['sample_sizes'].values())}). "
                    f"Interpret results with caution."
                )

            # Step 2: recommended test
            st.subheader("② Recommended test")
            st.success(f"**{TEST_DISPLAY_NAMES[test_name]}**")
            st.caption(diagnostics["decision_path"])
            if "alternative_test" in diagnostics:
                st.caption(
                    f"Alternative: {TEST_DISPLAY_NAMES[diagnostics['alternative_test']]} "
                    f"— distribution-free, robust for small or irregular samples"
                )

            # Step 3: run the test
            with st.spinner("Running the statistical test..."):
                result = run_test(test_name, group_a, group_b)

            st.subheader("③ Test results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Test statistic", result["statistic"])
            c2.metric("P-value", result["p_value"])
            c3.metric("Significant?", "Yes" if result["significant"] else "No")

            # Step 4: AI report
            with st.spinner(f"Generating report via {provider}..."):
                report = generate_report(
                    result, diagnostics,
                    context=context,
                    api_key=api_key or None,
                    provider=provider,
                )

            st.subheader("④ Report for stakeholders")
            st.write(report)
