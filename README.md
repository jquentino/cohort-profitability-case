# Financial AI Data Scientist Case Study: Profitability Under Uncertainty

## Overview

This case study tests your skills in data exploration, feature engineering, machine learning and analytics. You will work with a single SQLite `.db` containing the tables `allowlist`, `loans`, and `loans_repayments`.

Assume credit offers are extended to pre-approved cohorts (interchangeably, *batches*) defined by a common selection rule (model or heuristic) within a time window. Each new cohort is added to the `allowlist` table, that contains all the users that are eligible to take a loan.

Because rules and timing are shared, these cohorts exhibit consistent traits (merchant/product mix, pricing, underwriting policy, and seasonality) and they form an operational decision unit: if a cohort performs well, we scale the same criteria to future pre-approvals.

The goal is to determine whether a cohort is safely profitable by a chosen horizon. Outcomes depend on early repayment behavior, collections effectiveness, repeat-borrower dynamics, exposure concentration and other factors. Your objective is to consider these factors and estimate profitability, quantify the uncertainty around that estimate to support a defensible go/no-go decision.

## Objective

Analyze the provided data, understand the operational context, engineer meaningful features, and build a model that predicts ROI at a future horizon $H$ using only information available up to a specified decision time $t$ (assume $t=0$ is cohort creation). The model should leverage individual loan repayment histories and cohort information in the same form provided in the database.

## Table Schemas

### Allowlist Table
- **user_id**:  Unique identifier for the user.
- **batch**: Identifier of the cohort or experiment that allowed the user.
- **allowlisted_date**: Date on which the user was added to the allowlist (i.e., became eligible to receive a loan).

### Loans Table
- **loan_id**: Unique identifier for the loan.
- **user_id**:  Unique identifier for the user.
- **created_at**: Date on which the loan was created.
- **updated_at**: Date and time of the last update to the loan record (this may reflect a status change or modification of other columns).
- **annual_interest**: Annual interest rate (%) applied to the loan.
- **loan_amount**: Principal amount granted in the loan.
- **status**: Current loan status. Possible values that are not errors include:
    - **executed**: The loan has been disbursed to the user.
    - **repaid**: The loan was fully repaid within its regular term.
    - **debt_collection**: The loan was not repaid within its regular term and entered the collection process.
    - **debt_repaid**: The loan was eventually repaid after entering the debt collection stage.

### Loans Repayments Table
- **date**: Date of the repayment or billing event.
- **loan_id**: Unique identifier for the loan.
- **repayment_amount**: Amount repaid by the user on that date.
- **billings_amount**: Amount paid by the user during the collection process.

## Tasks

### 1) ROI-over-time EDA
Describe ROI as a function of time for each batch, using only information available up to each time point. Summarize key findings and use them later to motivate your evaluation horizon $H$.

The following optional lines of inquiry (you should develop and explore your own questions): 

- **Correlation of results:** If loans within the same cohort are predominantly being repaid, does this increase the chance that a particular loan will be repaid?
- **Sensitivity to large loans:** How much do top exposures move the ROI curve?
- **Collections sensitivity:** How would changes in billing amounts affect the curve?
- **First-loan tolerance:** How much initial default rate could this cohort tolerate and still reach non-negative ROI by horizon $H$?

### 2) Define decision time and horizon
Set the decision time $t$ (assume $t=0$ is batch creation) and choose the evaluation horizon $H$ for prediction. Justify your choice based on your ROI-over-time exploration.

> Your task is to predict ROI at horizon $H$ as a function of time-$t$ information and cohort characteristics.

### 3) Build decision-time representations
By using the provided tables, construct features available at time $t$ only.
- Include per-loan early-behavior signals.
- Include cohort-level composition/exposure features (e.g., value-weighted mix, ticket-size distribution, concentration indicators).

### 4) Choose a modeling strategy (you may try both)
- **(a) Loan-level → aggregate:** model per-loan outcomes and aggregate to the cohort, or
- **(b) Cohort-level:** model cohort outcomes directly using composition/exposure features.

In all cases, predict ROI$(H)$ given loan repayments, cohort information, and loan statuses in the same form they appear in the database.

### 5) Provide uncertainty around predictions
Report confidence intervals for your cohort-level $\mathrm{ROI}(H)$ predictions. Keep the method concise and appropriate to the data.

## Deliverables

1. **Exploratory Data Analysis (EDA):** A report or Jupyter notebook detailing your EDA process and findings.
2. **Feature Engineering Notebook:** Code and analysis for feature creation (SQL, Python, or any suitable approach).
3. **Modeling Notebook:** Data preprocessing, model training, and evaluation.
4. **`model_implementation.py`:** A file containing the model implementation used to generate predictions.
5. **Final Report:** A brief report outlining your approach, methodologies used, key insights, model performance, and recommendations. Don’t overthink this; we’ll review your code, but a short summary helps us review faster.
6. **src folder:** A folder containing the source code for your analysis, to aid the development and flow of the notebook and report.

## Evaluation Criteria
- **Quality of EDA:** Go beyond the obvious; produce insights that enable later steps.
- **Problem Structuring:** Clear framing of decision time, horizon, and constraints.
- **Feature Engineering:** Creativity and effectiveness of features available at time $t$.
- **Model Performance:** Accuracy and relevance of ROI predictions at $H$.
- **Clarity & Structure:** Well-organized, reproducible notebooks and reports.

## Tips for Success
- **Understand Relationships:** Verify how tables join and how entities relate; mistakes here cascade.
- **Document Decisions:** Record assumptions, choices, and rationale as you work.
- **Avoid Leakage:** Use only information available at or before time $t$ when creating features or labels.
- **Model Selection:** Start simple; for tabular data, tree-based models are often strong baselines. Any model is acceptable if justified and effective.
- **Address Limitations:** For any observed challenge, propose and document potential solutions, even if only partial or hypothetical.
- **Iterate:** Expect to refine earlier steps as new insights emerge.