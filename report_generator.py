"""
report_generator.py
====================
TrialStat AI — The only module that calls a Large Language Model (LLM).

Uses OpenAI-compatible API format throughout.
All models accessed via OpenRouter — one key, multiple free options.

Note: Free model availability on OpenRouter changes periodically.
If a model returns 404, check openrouter.ai/models and filter by "Free".
"""

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
2. A simple analogy or everyday comparison explaining why this conclusion follows
   (avoid raw statistical terms unless immediately explained in plain English).
3. One concrete, actionable business recommendation.
4. If the sample size is small or the result has notable limitations, flag this
   so the reader does not over-interpret the finding.

Do not use terms like Mann-Whitney, p-value < significance level, or
null hypothesis without pairing each with an immediate plain-English gloss.
"""

# Free models on OpenRouter — verified more stable options.
# If one stops working, try another or check openrouter.ai/models (filter: Free).
PROVIDERS = {
    "Nemotron 3 Ultra (free)": {
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "nvidia/nemotron-3-ultra-550b-a55b:free",
    },
    "Poolside: Laguna M.1 (free)": {
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "poolside/laguna-m.1:free",
    },
    "Nemotron 3 Super (free)": {
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "nvidia/nemotron-3-super-120b-a12b:free",
    },
    "gpt-oss-120b (free)": {
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "openai/gpt-oss-120b:free",
    },
}


def generate_report(test_result, diagnostics, context="",
                    api_key=None, provider="Nemotron 3 Ultra (free)"):
    """
    Call selected LLM via OpenRouter to generate a plain-language report.

    Parameters:
        test_result : dict from stat_tests.run_test()
        diagnostics : dict from decision_engine.select_test()
        context     : business background string
        api_key     : OpenRouter API key from sidebar
        provider    : key from PROVIDERS dict

    Returns:
        str — LLM-generated report, or structured fallback template
    """
    if not api_key:
        return _fallback_report(test_result, diagnostics)

    prompt = REPORT_PROMPT.format(
        context=context or "(No context provided)",
        test_name=test_result["test_name"],
        decision_path=diagnostics.get("decision_path", "not recorded"),
        statistic=test_result["statistic"],
        p_value=test_result["p_value"],
        significant="Yes" if test_result["significant"] else "No",
    )

    cfg = PROVIDERS.get(provider, PROVIDERS["Nemotron 3 Ultra (free)"])

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=cfg["base_url"],
            default_headers={
                "HTTP-Referer": "https://github.com/trialstat-ai",
                "X-Title": "TrialStat AI",
            },
        )
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            timeout=30,
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        # Give a clearer hint when it's a 404 free-model issue
        if "404" in error_msg and "free" in error_msg.lower():
            return (
                f"[This free model is currently unavailable on OpenRouter.]\n\n"
                f"Please try a different model from the sidebar dropdown, "
                f"or check openrouter.ai/models (filter: Free) for currently "
                f"available options.\n\n"
                f"Technical error: {error_msg}\n\n"
                + _fallback_report(test_result, diagnostics)
            )
        return _fallback_report(test_result, diagnostics, error=error_msg)


def _fallback_report(test_result, diagnostics, error=None):
    """Structured template shown when no API key is set or call fails."""
    lines = []
    if error:
        lines.append(f"[AI report generation failed — Error: {error}]\n")
    else:
        lines.append(
            "[No API key — showing structured template. "
            "Enter your OpenRouter key in the sidebar. "
            "Get a free key at openrouter.ai]\n"
        )
    lines.append(f"Test used      : {test_result['test_name']}")
    lines.append(f"Decision path  : {diagnostics.get('decision_path', 'not recorded')}")
    lines.append(f"Test statistic : {test_result['statistic']}")
    lines.append(f"P-value        : {test_result['p_value']}")
    if test_result["significant"]:
        lines.append("Conclusion     : The two groups show a statistically significant difference.")
    else:
        lines.append(
            "Conclusion     : The available data do not provide sufficient evidence "
            "of a difference between the two groups."
        )
    return "\n".join(lines)
