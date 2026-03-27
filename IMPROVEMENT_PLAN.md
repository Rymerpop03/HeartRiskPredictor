# Heart Disease Predictor — Improvement Plan

---

## Part 1: Model Accuracy Improvements

### 1. Fix Class Imbalance More Aggressively
- Your dataset has far fewer CHD-positive cases (minority class)
- Gradient Boosting is the only model without any imbalance handling right now
- **Action:** Add SMOTE (Synthetic Minority Oversampling) to the training pipeline, or add `class_weight='balanced'` to Gradient Boosting like the other models already have

### 2. Add Cross-Validation
- Currently you do a single 80/20 split — your results could be "lucky" or "unlucky" based on that one split
- **Action:** Add 5-fold or 10-fold stratified cross-validation so you get a reliable average performance score

### 3. Tune Hyperparameters
- All models are using hand-picked settings (e.g. learning rate 0.05, 200 trees, etc.)
- **Action:** Use `GridSearchCV` or `RandomizedSearchCV` to automatically find better settings for each model

### 4. Add Feature Engineering
- You only use the raw 15 features as-is
- **Action:** Create a few new combined features that doctors actually use, such as:
  - **Pulse pressure** = `sysBP - diaBP`
  - **Smoking intensity** = `currentSmoker × cigsPerDay`
  - **Metabolic risk score** = combine glucose + BMI + diabetes

### 5. Improve the Ensemble
- Right now the ensemble is a simple average of all 4 models equally
- **Action:** Use a **weighted ensemble** (give more weight to whichever model performs best on the validation set) or train a small "meta-model" (stacking)

### 6. Better Threshold Tuning
- All models currently use 0.5 as the decision cutoff (50% = positive prediction)
- For medical risk, **false negatives are worse than false positives** (missing a sick person is bad)
- **Action:** Lower the threshold (e.g. 0.35–0.40) to catch more true positives, and show the tradeoff curve in the analysis app

---

## Part 2: Predictor Interface Improvements

### 1. Add Explainability (SHAP Values)
- Users see a risk percentage but don't know *why*
- **Action:** Add a "Why is my risk this high?" section using SHAP that shows each feature's individual contribution (e.g. "Your glucose adds +8% risk")

### 2. Better Input Validation & Guidance
- Users might not know what "total cholesterol 200 mg/dL" means or if their value is normal
- **Action:** Add color-coded reference ranges next to each input (green/yellow/red) so users know if their value is concerning *before* they predict

### 3. Confidence Interval Display
- Right now you show a single number like "28% risk" with no indication of uncertainty
- **Action:** Show a range (e.g. "24% – 32%") using the spread between the 4 models as a simple uncertainty estimate

### 4. Risk Factor Priority Ranking
- The "Key Risk Factors" list shows all flags equally
- **Action:** Rank them by impact (e.g. "Your #1 modifiable risk factor is smoking") based on feature importance from the models

### 5. "What-If" Scenario Tool
- Users can't currently explore how lifestyle changes would affect their risk
- **Action:** Add a slider or toggle that lets users simulate: "What if I quit smoking?" or "What if I lower my BP by 10 mmHg?" and shows the new predicted risk instantly

### 6. Add History / Comparison
- Users have no way to track changes over time
- **Action:** Add a simple session-based log (or downloadable PDF report) so users can compare predictions across different inputs

---

## Part 3: Quick Wins (Low Effort, High Value)

| # | Improvement | Effort | Impact |
|---|-------------|--------|--------|
| 1 | Add `class_weight='balanced'` to Gradient Boosting | Very Low | Medium |
| 2 | Show confidence range using model spread | Low | High |
| 3 | Color-code input fields with normal ranges | Low | High |
| 4 | Add cross-validation to training scripts | Low | Medium |
| 5 | Lower prediction threshold for medical safety | Low | High |
| 6 | Add SHAP explainability to the UI | Medium | Very High |
| 7 | Add "What-If" scenario sliders | Medium | High |
| 8 | Hyperparameter tuning with RandomizedSearchCV | Medium | Medium |

---

## Priority Order

If starting from scratch on improvements, tackle these first:

1. **SHAP explainability** — biggest usability upgrade, makes the tool actually trustworthy
2. **Threshold tuning** — critical for a medical context
3. **Input range indicators** — easy to add, immediately useful
4. **Class imbalance fix for Gradient Boosting** — quick code change
5. **What-If simulator** — makes it a genuinely interactive tool
