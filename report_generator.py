"""
report_generator.py
====================
The only module in TrialStat AI that calls a Large Language Model (LLM).

Uses Google Gemini API (free tier: gemini-1.5-flash, 1500 requests/day).
Get your free API key at: https://aistudio.google.com

The preceding modules (decision_engine.py, stat_tests.py) are fully
deterministic — no AI involved. This module has one job: translate
statistical output into plain-language prose that a non-technical
stakeholder can act on.

Requires GEMINI_API_KEY environment variable for live generation.
Falls back to a structured template when the key is absent or the
API call fails, so the rest of the pipeline keeps running.

这是整个项目里唯一真正用到大语言模型（LLM）的部分。

前面的 decision_engine.py 和 stat_tests.py 全部是确定性的统计逻辑，
不依赖AI——该用什么检验、检验结果是多少，都是规则和数学算出来的，
可重复、可解释、不会彻底被AI决定我们的任何业务重点问题。

LLM在这里只做一件事：把复杂的统计数字（p值、统计量），
转译成业务方（比如本项目demo场景里的"农场主"）能听懂的人话报告。
这是语言生成任务，恰好是LLM擅长、而规则引擎做不到的部分。

需要设置环境变量 ANTHROPIC_API_KEY 才能实际调用。
没有API key时，会返回一个不调用AI的"基础报告"模板，方便测试核心逻辑。
"""

import os

REPORT_PROMPT = """\
You are a statistical consultant writing a brief report for a business stakeholder
who has no statistical background. Translate the technical results below into clear,
jargon-free language they can act on.

Context about the data:
{context}

Statistical test used: {test_name}
Why this test was chosen: {decision_path}
Test statistic: {statistic}
P-value: {p_value}
Significance level: 0.05
Result is statistically significant: {significant}

Write 150-250 words in the following structure:
1. One sentence stating the conclusion — is there evidence of an effect or not?
2. A simple analogy or everyday comparison explaining *why* this conclusion follows
   (avoid raw statistical terms unless immediately explained in plain English).
3. One concrete, actionable business recommendation.
4. If the sample size is small or the result has notable limitations, flag this
   so the reader does not over-interpret the finding.

Do not use terms like "Mann-Whitney", "p-value < significance level", or
"null hypothesis" without pairing each with an immediate plain-English gloss.
"""


def generate_report(test_result, diagnostics, context=""):
    """
    Call the Gemini API to generate a plain-language business report.

    Parameters:
        test_result  : dict returned by stat_tests.run_test()
        diagnostics  : dict returned by decision_engine.select_test()
        context      : str describing the business background of this dataset

    Returns:
        str — the generated report (LLM output or fallback template)
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    prompt = REPORT_PROMPT.format(
        context=context or "(No context provided)",
        test_name=test_result["test_name"],
        decision_path=diagnostics.get("decision_path", "not recorded"),
        statistic=test_result["statistic"],
        p_value=test_result["p_value"],
        significant="Yes" if test_result["significant"] else "No",
    )

    if not api_key:
        return _fallback_report(test_result, diagnostics)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # gemini-1.5-flash: free tier, 1500 requests/day, no credit card needed
        model    = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return _fallback_report(test_result, diagnostics, error=str(e))


def _fallback_report(test_result, diagnostics, error=None):
    """Structured template used when LLM generation is unavailable."""
    lines = []
    if error:
        lines.append(
            f"[AI report generation failed — falling back to template.\n"
            f"Error: {error}]\n"
        )
    else:
        lines.append(
            "[GEMINI_API_KEY not set — showing structured template.\n"
            " Get a free key at https://aistudio.google.com]\n"
        )

    lines.append(f"Test used      : {test_result['test_name']}")
    lines.append(f"Decision path  : {diagnostics.get('decision_path', 'not recorded')}")
    lines.append(f"Test statistic : {test_result['statistic']}")
    lines.append(f"P-value        : {test_result['p_value']}")

    if test_result["significant"]:
        lines.append(
            "Conclusion     : The two groups show a statistically significant difference."
        )
    else:
        lines.append(
            "Conclusion     : The available data do not provide sufficient evidence "
            "of a difference between the two groups."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    from decision_engine import select_test
    from stat_tests import run_test

    control   = [22.5, 23, 31.5, 23.5]
    treatment = [21.5, 0.75, 4.3, 30, 3, 28.5, 11.5, 24.5]

    test_name, diagnostics = select_test(control, treatment)
    result = run_test(test_name, control, treatment)

    report = generate_report(
        result, diagnostics,
        context=(
            "A farmer wants to know whether a deworming vaccine effectively "
            "reduced the number of parasites in vaccinated goats compared to "
            "unvaccinated controls."
        ),
    )
    print(report)
