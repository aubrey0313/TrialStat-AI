"""
app.py
======
Streamlit web interface for TrialStat AI.

Orchestrates the full pipeline:
  Upload CSV -> select columns -> assumption checks -> test recommendation
  -> run test -> AI report generation

Run with:
    streamlit run app.py


Streamlit网页界面，把整个流程串起来：

  上传CSV → 选择分组 → 自动判断统计假设 → 推荐检验方法
  → 执行检验 → AI生成业务报告

运行方式：
    streamlit run app.py

需要先 pip install -r requirements.txt
"""

import streamlit as st
import pandas as pd

from decision_engine import select_test, TEST_DISPLAY_NAMES
from stat_tests import run_test
from report_generator import generate_report

st.set_page_config(
    page_title="TrialStat AI",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 TrialStat AI")
st.caption(
    "Automated Clinical Trial Statistical Analysis · "
    "Upload two-group experimental data · Automatic test selection · "
    "Plain-language business report"
)

with st.expander("How does this work?", expanded=False):
    st.markdown("""
    **Statistical decision (rule engine — no AI here)**  
    The tool runs Shapiro-Wilk normality tests and Levene's variance homogeneity
    test on your data, then applies a deterministic decision tree to recommend
    the most appropriate test:
    Independent t-test / Welch's t-test / Mann-Whitney U (+ Bootstrap as alternative).

    **Report generation (LLM — only this step uses AI)**  
    The statistical output is passed to a language model, which translates
    the numbers into plain-language findings and recommendations your
    stakeholders can act on.
    """)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a CSV file (one column for group labels, one for numeric values)",
    type="csv",
)

use_demo = st.button("No data? Try the goat deworming vaccine demo dataset")

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_demo:
    df = pd.read_csv("sample_data/goat_worms.csv")
    st.info(
        "Demo data loaded: goat deworming vaccine trial "
        "(control group vs. vaccinated group)"
    )

if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        group_col = st.selectbox("Group column", df.columns, index=0)
    with col2:
        value_col = st.selectbox(
            "Value column", df.columns,
            index=min(1, len(df.columns) - 1)
        )

    groups = df[group_col].dropna().unique()

    if len(groups) != 2:
        st.error(
            f"Column '{group_col}' contains {len(groups)} unique groups. "
            f"TrialStat AI compares exactly two independent groups — "
            f"please check your data."
        )
    else:
        label_a, label_b = groups[0], groups[1]
        group_a = df[df[group_col] == label_a][value_col].dropna().values
        group_b = df[df[group_col] == label_b][value_col].dropna().values

        context = st.text_area(
            "Business context (optional — personalises the AI report)",
            placeholder=(
                "e.g. A pharmaceutical company wants to know whether a new "
                "deworming vaccine reduces parasite counts in goats compared "
                "to unvaccinated controls..."
            ),
        )

        if st.button("Run analysis", type="primary"):

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
                    f"⚠️ Small sample detected "
                    f"(minimum group size: "
                    f"{min(diagnostics['sample_sizes'].values())}). "
                    f"Interpret results with caution."
                )

            st.subheader("② Recommended test")
            st.success(f"**{TEST_DISPLAY_NAMES[test_name]}**")
            st.caption(diagnostics["decision_path"])

            if "alternative_test" in diagnostics:
                st.caption(
                    f"Alternative: "
                    f"{TEST_DISPLAY_NAMES[diagnostics['alternative_test']]} "
                    f"— distribution-free, especially robust for small or "
                    f"irregular samples"
                )

            with st.spinner("Running the statistical test..."):
                result = run_test(test_name, group_a, group_b)

            st.subheader("③ Test results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Test statistic", result["statistic"])
            c2.metric("P-value", result["p_value"])
            c3.metric("Significant?", "Yes" if result["significant"] else "No")

            with st.spinner("Generating plain-language report..."):
                report = generate_report(result, diagnostics, context=context)

            st.subheader("④ Report for stakeholders")
            st.write(report)
