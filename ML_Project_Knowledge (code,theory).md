# Cancer Risk Prediction Notebook — Full Teaching Walkthrough

This is your notebook explained cell-by-cell, concept-by-concept. Read it next
to the actual `.ipynb` so you can match explanations to code.

---

## 0. Setup — libraries

```python
!pip install -q optuna imbalanced-learn xgboost joblib kagglehub
```

Colab comes with pandas/sklearn pre-installed but not everything. We install:
- **optuna** → automatic hyperparameter search
- **imbalanced-learn** → gives us `SMOTE`
- **xgboost** → the gradient boosting library
- **joblib** → saves Python objects (models, encoders) to `.pkl` files
- **kagglehub** → optional, for pulling the dataset straight from Kaggle

Then we import sklearn pieces for splitting data, encoding, modeling, and
scoring — these get reused throughout, so it's worth knowing what each import
is *for*, not just running the cell blindly.

**Core idea for this whole notebook:** ML models only understand numbers. Every
step from here on is either (a) turning raw data into clean numbers, (b)
training a model on those numbers, or (c) measuring how good the model is.

---

## 0.1 Loading the data

We read the CSV into a pandas DataFrame — think of it as an Excel sheet living
in memory. `df_raw.info()` and `.isnull().sum()` are your first diagnostic
tools:
- `.info()` tells you the data type of every column (numbers vs text) and how
  many non-null values it has.
- `.isnull().sum()` counts missing values per column — important because most
  models can't handle blanks directly.

## 0.2 Finding the target column

The **target** is the column you're trying to predict — here, `Risk_Level`.
Everything else is a **feature** (an input the model uses to make its
prediction). We auto-detect the target column name defensively, since real
datasets often have slightly different header spellings than you expect.

`value_counts()` + the countplot show the **class distribution** — how many
Low/Medium/High rows exist. This one plot explains almost everything that
happens later in the notebook: if one class (usually "High") has far fewer
rows than the others, the model will naturally get lazy and under-predict it.
That's called **class imbalance**, and steps 5–10 exist specifically to fight it.

---

## Step 1 — Basic Models (Logistic Regression & Random Forest)

### Why start with a "bad" baseline on purpose?
You always want a simple reference point. If your fancy tuned model later gets
84% accuracy, is that good? You can't know without comparing it to something
simple first. Baselines also reveal problems early (like leakage — see Step 2).

### `encode_features()` — turning text into numbers
Models can't read the word `"Male"` or `"High"` — they need numbers. This
function:
1. Takes the target column (e.g. `Risk_Level`) and turns `Low/Medium/High`
   into `0/1/2` using a `LabelEncoder`. It **saves** this encoder so later we
   can convert predictions (`0,1,2`) back into human-readable labels.
2. Does the same thing for every text-based (categorical) feature column,
   e.g. `Gender: Male/Female` → `0/1`.

We keep the fitted encoders in a dictionary because **the exact same
encoding has to be reused at prediction time** — that's literally what
`label_encoders.pkl` is for in your Streamlit app.

### Train/test split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, ...)
```
- `X` = features, `y` = target.
- We train on 80% of the data and **never let the model see the other 20%**
  during training — that 20% (`X_test`) simulates "new patients" so we get an
  honest read on how well the model generalizes, instead of just memorizing
  the training data.
- `stratify=y` keeps the Low/Medium/High ratio the same in both the train and
  test sets — otherwise a random split might accidentally put almost all the
  "High" cases into one side.

### Logistic Regression vs Random Forest
- **Logistic Regression**: draws (roughly) straight decision boundaries
  between classes. Fast, interpretable, but weak when relationships in the
  data are non-linear or involve complex feature interactions.
- **Random Forest**: builds many decision trees, each trained on a random
  subset of data/features, and averages their votes. Handles non-linear
  patterns and feature interactions much better, at the cost of being less
  interpretable.

### Reading the output
`classification_report` gives you, per class:
- **Precision** — of everything the model called "High," what % actually was High?
- **Recall** — of everything that actually was High, what % did the model catch?
- **F1-score** — the harmonic mean of precision and recall (a balanced summary).

For a health-risk problem, **recall on the High class is usually the number
you care about most** — missing a real high-risk patient is worse than a false
alarm. Keep that in mind; it's why later steps optimize for recall, not just
accuracy.

---

## Step 2 — Data Leakage Fix

### What is leakage?
**Leakage** = accidentally giving the model a feature that already contains
(or is derived from) the answer. If `Overall_Risk_Score` is basically a
formula that already computed the risk level, the model doesn't need to learn
anything — it just reads the answer off that column. You'd get suspiciously
high accuracy in testing, then the model would fail in the real world where
that pre-computed score doesn't exist yet (because that's literally what
you're trying to predict!).

### How we detect it
```python
leak_candidates = [c for c in df_raw.columns if "risk" in c.lower() and "score" in c.lower()]
```
A simple heuristic: any column name containing both "risk" and "score" is
almost certainly a derived summary column, not a raw feature. We drop it and
re-encode everything.

**Rule of thumb for your own future projects:** always ask "would this
column exist *before* I know the answer, in the real world?" If not, it's
leakage — drop it.

---

## Step 3 — Random Forest (leakage-free)

Same model as Step 1, but now trained honestly. Two new tools introduced here:

### Confusion Matrix
A grid showing actual class (rows) vs predicted class (columns). The diagonal
= correct predictions. Anything off-diagonal is a specific kind of mistake
(e.g., "actually High, predicted Medium"). This is far more informative than a
single accuracy number because it shows *which* classes get confused with
which.

### Feature Importance
Random Forest can tell you which columns it relied on most to make decisions.
This is useful for two reasons: (1) sanity-checking the model (do the top
features make medical/behavioral sense?), and (2) catching leakage you might
have missed — if some obscure column is suspiciously the #1 most important
feature, investigate it.

---

## Step 4 — Hyperparameter Tuning with Optuna

### What's a hyperparameter?
A setting you choose *before* training that controls how the model learns —
e.g., how many trees in the forest, how deep each tree can grow. These are
different from the model's learned parameters (which it figures out from
data on its own).

### Why not just guess good values?
You could try `n_estimators=100`, then `200`, then `300`... manually — that's
slow and unscientific. **Optuna** automates this: it runs many trials, each
time picking new hyperparameter values (intelligently, based on what worked
in earlier trials — this is called Bayesian optimization), trains a model,
scores it, and remembers the best combination.

### The `objective` function
```python
def rf_objective(trial):
    params = {...trial.suggest_int(...), trial.suggest_categorical(...)}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds, average="macro")
```
This is the function Optuna calls repeatedly. `trial.suggest_int(...)` means
"try a value in this range." We return **macro F1** (F1 averaged equally
across all 3 classes) rather than accuracy — this stops Optuna from just
learning to always predict the majority class to maximize accuracy while
ignoring the minority "High" class.

`study.optimize(objective, n_trials=30)` runs the whole search 30 times and
keeps the best result in `study.best_params`.

---

## Step 5 — Class Balancing with SMOTE

### The core problem
If 70% of your training rows are "Low" risk and only 10% are "High" risk, a
model can get "decent" accuracy just by mostly ignoring High risk and
guessing Low most of the time. That's obviously bad for a medical tool.

### What SMOTE actually does
**SMOTE (Synthetic Minority Over-sampling Technique)** creates new, synthetic
"High risk" training examples by interpolating between real High-risk
patients that are similar to each other in feature space — it's not
duplicating rows, it's generating plausible new ones along the line between
existing minority-class neighbors.

### Critical rule
```python
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)  # only on TRAIN
```
We only apply SMOTE to the **training** set, never the test set. The test set
must stay 100% real data, or your evaluation becomes fake — you'd essentially
be grading the model on synthetic answers it partly generated itself.

---

## Step 6 — Focus on High-Risk Recall

This step doesn't introduce new modeling — it's a **discipline** step. We
specifically isolate: out of all real High-risk patients in the test set, how
many did the model catch? This reframes the whole project around the metric
that actually matters clinically, instead of a generic "accuracy" number that
can hide poor minority-class performance.

---

## Step 7 — XGBoost with SMOTE

### Random Forest vs XGBoost (Gradient Boosting)
- **Random Forest** builds many trees **independently** (in parallel) and
  averages their votes.
- **XGBoost** builds trees **sequentially** — each new tree is trained
  specifically to fix the mistakes of the trees before it. This usually
  produces stronger, more accurate models, especially on structured/tabular
  data like this dataset, though it's more prone to overfitting if not
  tuned carefully (hence Optuna tuning it in Step 8).

We train it on the same SMOTE-balanced data from Step 5 and compare results.

---

## Step 8 — Optuna + XGBoost for High Recall

Same idea as Step 4, but two upgrades:
1. **More hyperparameters** relevant to XGBoost specifically — `learning_rate`
   (how much each new tree corrects previous mistakes), `max_depth`, `gamma`
   (controls tree pruning/complexity), etc.
2. **A custom objective that blends two metrics:**
```python
return 0.5 * macro_recall + 0.5 * high_recall
```
This explicitly tells Optuna: "I don't just want overall balance — I want you
to specifically favor hyperparameters that catch more real High-risk
patients." This is the difference between optimizing for a textbook metric
and optimizing for what the project is actually for.

---

## Step 9 — Class-Weighted XGBoost (No SMOTE)

### An alternative to synthetic data
Instead of inventing new High-risk rows (SMOTE), we can instead tell the
model "when you get a High-risk example wrong during training, that mistake
counts more" — via **sample weights**.

```python
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)
```
`class_weight="balanced"` automatically gives rarer classes higher weight
(inversely proportional to how frequent they are). The big advantage: **the
model only ever learns from real patient data**, never synthetic rows —
often a safer choice for something like a health-risk model, since SMOTE's
synthetic points can sometimes sit in unrealistic regions of feature space.

---

## Step 10 — Final Model: Optuna-Tuned Class-Weighted XGBoost

This combines the best idea from each earlier step:
- **Step 9's class weighting** (real data only, rarer classes weighted higher)
- **Step 8's Optuna search** (systematically find the best hyperparameters)
- **A blended objective** of macro F1 and High-risk recall, plus this time
  two new hyperparameters: `reg_alpha` and `reg_lambda` — these are
  **regularization** terms that penalize overly complex trees, helping the
  model generalize instead of memorizing training data (overfitting).

The comparison table/bar chart at the end is good practice for any ML
project: **always show your work** — a reviewer (or future you) should be
able to see the whole journey from "dumb baseline" to "final model," not just
the final number.

---

## Step 11 — Exporting Artifacts

### Why `.pkl` files?
A trained model is just a Python object sitting in memory — it disappears
when the Colab runtime shuts down. `joblib.dump(obj, "file.pkl")` serializes
(saves) that object to disk so it can be loaded again later, in a completely
different program (your Streamlit app), without retraining.

We save **four** objects, and it's worth understanding why each one is
necessary at prediction time:
| File | Why it's needed |
|---|---|
| `model.pkl` | The actual trained XGBoost model |
| `label_encoders.pkl` | To convert a new patient's "Male"/"Female" into the same 0/1 the model was trained on |
| `target_encoder.pkl` | To convert the model's raw prediction (0/1/2) back into "Low"/"Medium"/"High" |
| `feature_info.pkl` | Column order + valid ranges/categories, so the Streamlit form can be built automatically and the input row can be assembled in the exact order the model expects |

**Important detail:** the column order fed into `model.predict()` must
*exactly* match the order used during training — that's why we explicitly
store and reuse `feature_order`.

---

## The ID-column fix (your recent question)

### Why identifiers are dangerous, not just useless
An ID column like `Patient_ID` is usually random/sequential and shouldn't
carry real medical signal — but a powerful model like XGBoost can sometimes
find spurious correlations with it anyway (e.g., if IDs were assigned in the
order patients were admitted, and admission order happens to correlate with
something). That's a subtle form of leakage/overfitting, on top of it just
being pointless clutter in a UI.

### The detection trick
```python
cardinality_flag = df_raw[c].nunique(dropna=True) / n_rows > 0.9
```
This is a **data-driven** (not just name-based) way to catch identifier
columns: if a column has a nearly-unique value for almost every row, it's
behaving like an ID regardless of what it's called. This is a generally
useful pattern to remember for any future tabular ML project.

---

## The big-picture lesson

Every step in this notebook follows the same loop, which is basically the
whole practice of applied ML:

1. **Train** a model
2. **Evaluate** it honestly (on data it never saw)
3. **Diagnose** what's wrong (leakage? imbalance? bad metric choice?)
4. **Fix** the specific problem you found
5. Repeat, always comparing back to your last result

Accuracy alone almost never tells the full story — that's why this project
moves from accuracy → confusion matrix → per-class recall → a custom blended
metric, getting more precise about "what does 'good' actually mean here" at
each stage.
