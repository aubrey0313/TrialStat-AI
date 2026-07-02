# TrialStat AI — Automated Clinical Trial Statistical Analysis Tool

An AI-powered pipeline for experimental data analysis:  
Upload two-group trial data → automatically select the appropriate statistical test → run the test → generate a plain-language report for non-statistical stakeholders.

## Motivation

This project originated from a statistical consulting engagement where I analysed a goat deworming vaccine trial dataset for a pharmaceutical client: determining the appropriate statistical test, explaining the statistical concepts, and delivering actionable business recommendations. I then formalised that manual workflow into a reusable automated tool.

## Core Design Philosophy: Separating Rule Logic from AI

> I did **not** let the language model decide which statistical test to use — a wrong choice here can directly mislead business decisions. The LLM is reserved for the step where its errors carry the lowest cost: translating numbers into readable prose.

| Module | Nature | Rationale |
|---|---|---|
| `decision_engine.py` Statistical assumption checking | **Deterministic rule logic — not AI** | The choice of test directly affects business decisions. The logic must be explainable, reproducible, hallucination-free, and consistent with both statistical theory and real-world consulting practice. |
| `stat_tests.py` Test computation | **Pure mathematics** | scipy statistics library; results are verifiable and reproducible. |
| `report_generator.py` Report generation | **Large language model (LLM)** | Translating statistical output into business language is a text-generation task — exactly where LLMs excel. Even if the prose is imperfect, it cannot corrupt the statistical conclusion itself. |

## Decision Logic

```
Two independent sample groups
          │
          ▼
  Normality test (Shapiro-Wilk)
          │
     ┌────┴────┐
  Non-normal  Normal
     │            │
     ▼            ▼
Mann-Whitney   Variance homogeneity test (Levene's)
U test                  │
(alt: Bootstrap)   ┌────┴────┐
               Equal var  Unequal var
                   │            │
                   ▼            ▼
        Independent t-test   Welch's t-test
```

When the smaller group has fewer than **5 observations**, the pipeline conservatively routes to Mann-Whitney U regardless of normality, and flags Bootstrap resampling as an alternative.

## Quick Start

```bash
# Step 1 — install dependencies (run once only)
pip install -r requirements.txt

# Step 2 — optional: set API key to enable AI report generation
export ANTHROPIC_API_KEY="your_key_here"

# Step 3 — launch the app (browser opens automatically)
streamlit run app.py
```

Without an API key the tool still runs fully — statistical assumption checks and test computations work normally; the report step falls back to a structured template.

## Validation

The bundled demo data (`sample_data/goat_worms.csv`) comes from the consulting engagement that inspired this project. The original SPSS analysis yielded Mann-Whitney p = 0.283; this project's Python implementation yields p = 0.2828 — confirming the core logic is correct.

## Project Structure

```
TrialStat-AI/
├── app.py                 # Streamlit web interface
├── decision_engine.py     # Assumption checks + test recommendation (rule engine)
├── stat_tests.py          # Four statistical test implementations
├── report_generator.py    # LLM-powered plain-language report generation
├── requirements.txt       # Python dependencies
└── sample_data/
    └── goat_worms.csv     # Demo dataset (goat deworming vaccine trial)
```

## Future Extensions

- Multi-group comparison (ANOVA / Kruskal-Wallis)
- Paired-sample test support
- Automatic PDF report export
- Effect size calculation (beyond p-values)

---

# TrialStat AI — 临床试验数据统计检验自动化工具

AI驱动的实验数据分析流水线：  
上传两组实验数据 → 自动判断合适的统计检验方法 → 执行检验 → 生成给非统计背景业务方看的人话报告。

## 项目动机

这个项目源于我在统计咨询工作中完成的一份任务——根据药企客户提供的山羊驱虫疫苗实验数据，判断该用什么统计检验、解释统计概念、给出业务建议。我把当时的人工分析流程，转化成了一套可复用的自动化工具。

## 核心设计理念：分清"规则逻辑"和"AI"的边界

> 我没有让大模型来决定该用什么统计检验，因为这种判断如果出错可能误导业务决策；我把大模型留给了它真正擅长、出错代价最低的环节——结果解释和报告撰写。

| 模块 | 性质 | 为什么这么设计 |
|---|---|---|
| `decision_engine.py` 统计假设判断 | **确定性规则逻辑，不是AI** | 该用哪种检验方法直接影响业务决策的正确性，这种判断需要可解释、可重复、不能有"幻觉"风险，并且得同时符合统计学以及业务落地的逻辑，所以用统计学规则而非大模型来做 |
| `stat_tests.py` 检验计算 | **纯数学计算** | scipy统计库，结果可验证、可复现 |
| `report_generator.py` 报告生成 | **大语言模型（LLM）** | 把统计数字转译成业务语言是语言生成任务，恰好是LLM的强项，且就算输出有偏差，也不会影响统计结论本身的正确性 |

## 决策逻辑

```
两组独立样本数据
      │
      ▼
 正态性检验 (Shapiro-Wilk)
      │
   ┌──┴──┐
不满足正态  满足正态
   │        │
   ▼        ▼
Mann-Whitney  方差齐性检验 (Levene's test)
U 检验           │
（备选：       ┌──┴──┐
Bootstrap）   方差齐   方差不齐
                │        │
                ▼        ▼
          独立样本t检验  Welch's t检验
```

样本量过小（默认阈值：单组少于 **5** 个观测值）时，无论正态性如何，都会保守地走向 Mann-Whitney U 检验路径，并提示 Bootstrap 重抽样法作为备选。

## 快速开始

```bash
# 第一步：安装依赖（只需做一次）
pip install -r requirements.txt

# 第二步：可选，配置 API key 启用 AI 报告生成
export ANTHROPIC_API_KEY="你的key"

# 第三步：启动项目（浏览器自动弹开）
streamlit run app.py
```

没有配置 API key 也能运行——统计判断和检验计算照常工作，报告部分会降级为基础模板。

## 验证

项目自带的demo数据（`sample_data/goat_worms.csv`）来自统计咨询工作，原分析用SPSS计算的Mann-Whitney检验得到 p = 0.283，本项目用Python重新计算得到 p = 0.2828，结果一致，验证了核心逻辑的正确性。

## 项目结构

```
TrialStat-AI/
├── app.py                 # Streamlit网页界面
├── decision_engine.py     # 统计假设判断 + 检验方法推荐（规则引擎）
├── stat_tests.py          # 四种检验方法的实际计算
├── report_generator.py    # 调用LLM生成业务报告
├── requirements.txt       # Python依赖清单
└── sample_data/
    └── goat_worms.csv     # demo数据（山羊驱虫疫苗实验）
```

## 后续可扩展方向

- 支持两组以上的多组比较（ANOVA / Kruskal-Wallis）
- 支持配对样本检验
- 自动生成PDF格式报告
- 支持效应量（effect size）计算，不只是看p值
