# 💳 Financial Risk Analysis for Loan Approval

> **AI Engineer Role — Pre-Work Assignment | Option 1**  
> Submitted by: **Karishma Patil** | ML Engineer | Bangalore  
> 📧 pkarishma8989@gmail.com | 📱 9172935283

---

## 📁 Repository Structure

```
├── README.md                        # This file — full assignment writeup
├── loan_risk_model.py               # Core ML pipeline with critical function walkthrough
├── requirements.txt                 # Dependencies
└── prework_assignment.pdf           # PDF version of this submission
```

---

## Section 1: Context

### Project Description

This project involved building an end-to-end machine learning pipeline to assess **credit risk** and automate **loan approval decisions** for a banking application. The system predicts two outputs: (1) whether a loan application should be approved (classification), and (2) the estimated loan amount to disburse (regression). The solution was deployed via a **Flask REST API on AWS EC2**, allowing the bank's internal systems to get real-time predictions on customer applications.

### Primary Technical Constraints

| Constraint | Details |
|---|---|
| **Class Imbalance** | Approved vs. rejected loans were heavily skewed (~80:20), requiring SMOTE oversampling |
| **Missing Data** | ~15–20% of customer records had missing income, credit history, and employment fields |
| **Latency** | API response time needed to stay under 500ms for real-time approval decisions |
| **Interpretability** | Banking domain requires explainable predictions — black-box models were discouraged |
| **Data Privacy** | Customer PII had to stay within AWS infrastructure; no external API calls with raw data |

---

## Section 2: Technical Implementation

### Architecture Diagram

```
[ Customer Application ]
         │
         ▼
[ Flask REST API (AWS EC2) ]
         │
         ▼
[ Preprocessing Pipeline ]
   ├── Missing Value Imputation (KNN / Regression-based)
   ├── Outlier Treatment (IQR / Z-score)
   ├── SMOTE Oversampling (train set only)
   └── Feature Scaling (StandardScaler)
         │
         ▼
┌────────────────────────────────┐
│   Dual Model Inference         │
│  ┌──────────────────────────┐  │
│  │ Classifier (Random Forest│  │
│  │ Loan Approved? Yes / No  │  │
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │ Regressor (Random Forest)│  │
│  │ Estimated Loan Amount    │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
         │
         ▼
[ JSON Response → Banking System ]
         │
         ▼
[ Logs stored in AWS S3 ]
```

**Explanation:** Raw customer data enters through a Flask API endpoint, passes through a shared preprocessing pipeline, and feeds into two separate Random Forest models — one for approval classification and one for loan amount regression. Results are returned as a structured JSON response and logged to S3.

### Code Walk-Through: Critical Function

See `loan_risk_model.py` for full implementation. The most critical function is `preprocess_and_predict()`:

```python
def preprocess_and_predict(raw_input: dict) -> dict:
    """
    End-to-end preprocessing and dual-model prediction for a single loan application.

    Steps:
    1. Convert input dict to DataFrame
    2. Impute missing values using trained KNN imputer
    3. Treat outliers using IQR-based clipping
    4. Scale features using fitted StandardScaler
    5. Run classification model → approval decision
    6. If approved, run regression model → loan amount estimate
    7. Return structured prediction result
    """
    df = pd.DataFrame([raw_input])

    # Step 1: Impute missing values using KNN (k=5, trained on training data)
    df_imputed = pd.DataFrame(
        knn_imputer.transform(df),
        columns=df.columns
    )

    # Step 2: Clip outliers using IQR bounds computed during training
    for col in NUMERIC_COLS:
        df_imputed[col] = df_imputed[col].clip(
            lower=iqr_bounds[col]['lower'],
            upper=iqr_bounds[col]['upper']
        )

    # Step 3: Scale features
    df_scaled = scaler.transform(df_imputed)

    # Step 4: Classification — is the loan approved?
    approval_proba = clf_model.predict_proba(df_scaled)[0][1]
    approval_decision = int(approval_proba >= APPROVAL_THRESHOLD)  # threshold=0.5

    # Step 5: Regression — only estimate amount if approved
    loan_amount = None
    if approval_decision == 1:
        loan_amount = float(reg_model.predict(df_scaled)[0])

    return {
        "approved": bool(approval_decision),
        "approval_probability": round(float(approval_proba), 4),
        "estimated_loan_amount": round(loan_amount, 2) if loan_amount else None
    }
```

**Why this is the most critical function:** Every API call flows through this function. It handles imputation, outlier treatment, scaling, and dual-model inference in a single pass — meaning any bug here directly impacts production decisions.

### Data Flow: Loan Approval Prediction

```
[Raw JSON Input]
    │
    ├── Validate required fields (age, income, credit_score, employment_years, etc.)
    │
    ├── KNN Imputer → fills missing numeric values based on 5 nearest neighbors
    │
    ├── IQR Clipping → caps outliers at [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    │
    ├── StandardScaler → normalizes features to mean=0, std=1
    │
    ├── Random Forest Classifier → P(approved) score
    │       └── If P >= 0.5 → Approved
    │       └── If P < 0.5  → Rejected
    │
    ├── Random Forest Regressor (only if approved) → Loan Amount estimate
    │
    └── [JSON Response: approved, probability, loan_amount]
```

---

## Section 3: Technical Decisions

### Decision 1: Random Forest over Logistic Regression

| | Random Forest | Logistic Regression |
|---|---|---|
| **Accuracy** | 85% | 79% |
| **Handles Non-linearity** | ✅ Yes | ❌ No |
| **Interpretability** | Medium (feature importance) | High (coefficients) |
| **Training Speed** | Slower | Fast |
| **Overfitting Risk** | Low (ensemble) | Low |

**Why I chose Random Forest:** The customer dataset had non-linear relationships between features (e.g., income × credit score interaction). Logistic Regression struggled to capture this, giving 79% accuracy vs. RF's 85%. For the banking domain, I also used feature importance scores from RF to provide partial explainability to stakeholders.

**Trade-off accepted:** Random Forest is slower to train and less interpretable than Logistic Regression. For real-time inference, I mitigated this by pre-loading the serialized model (joblib) at Flask startup — keeping inference time under 100ms.

---

### Decision 2: KNN Imputation over Mean/Median Imputation

| | KNN Imputation | Mean/Median Imputation |
|---|---|---|
| **Preserves Relationships** | ✅ Yes | ❌ No |
| **Handles Non-random Missingness** | ✅ Better | ❌ Weak |
| **Computation Cost** | Higher | Very Low |
| **Impact on Model Accuracy** | +2–3% improvement | Baseline |

**Why I chose KNN Imputation:** Customer data had Missing Not At Random (MNAR) patterns — for example, self-employed customers were more likely to leave income fields blank. Mean imputation would have introduced bias. KNN imputer fills values based on similar customer profiles, preserving feature correlations.

**Trade-off accepted:** KNN imputation is computationally heavier. At inference time this was negligible (single row), but during batch retraining on 50k+ records, it added ~3 minutes. Acceptable given the accuracy improvement.

---

### Scaling Bottleneck & Mitigation

**Bottleneck identified:** The Flask API was single-threaded and synchronous. Under concurrent loan application submissions (e.g., 100+ requests/second during peak hours), response times spiked from 80ms to 4+ seconds.

**Mitigation Strategy:**
1. **Gunicorn with multiple workers** — switched from Flask dev server to Gunicorn with 4 async workers on EC2
2. **Model caching** — loaded models into memory once at startup using `joblib.load()`, not per request
3. **AWS Auto Scaling** — configured EC2 Auto Scaling group to spin up additional instances when CPU > 70%
4. **Future plan:** Move to AWS SageMaker endpoints which handle autoscaling natively

---

## Section 4: Learning & Iteration

### One Technical Mistake

**Mistake: Applied SMOTE before train-test split**

Early in the project, I applied SMOTE oversampling to the entire dataset before splitting into train/test sets. This caused **data leakage** — synthetic samples from the training distribution were present in the test set, making the model appear to perform better than it actually did (inflated accuracy from ~82% to ~89%).

**What I learned:** Always split first, then apply oversampling **only on the training set**. The test set must represent real-world, unseen data — never augmented or synthetic. I now use `Pipeline` from Scikit-learn to enforce this ordering.

```python
# WRONG (what I did initially)
X_resampled, y_resampled = smote.fit_resample(X, y)  # leakage!
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

# CORRECT (fixed approach)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  # only on train
```

---

### One Thing I'd Do Differently Today

**I would add model monitoring and drift detection from day one.**

After deployment, the model's performance gradually degraded as customer loan patterns shifted (economic changes, new customer segments). We had no alerts set up, so we only noticed when business stakeholders flagged unusual approval rates weeks later.

Today, I would integrate **Evidently AI** or **AWS Model Monitor** to track:
- Feature distribution drift (PSI score)
- Prediction distribution shifts
- Model accuracy on a rolling window of labeled data

This would trigger automatic retraining or at minimum an alert to the ML team within 24 hours of significant drift.

---

## 📊 Results Summary

| Metric | Score |
|---|---|
| Classification Accuracy | **85%** |
| AUC-ROC | **0.91** |
| Regression R² Score | **0.82** |
| MAE (Loan Amount) | ~₹12,000 |
| API Response Time | < 100ms |

---

## 🛠️ Tech Stack

`Python` · `Scikit-learn` · `Pandas` · `NumPy` · `Flask` · `AWS EC2` · `AWS S3` · `joblib` · `SMOTE (imbalanced-learn)` · `GridSearchCV` · `Random Forest` · `Logistic Regression`

---

*Thank you for reviewing my submission! Happy to walk through any section in more depth during the next round.* 🙏
