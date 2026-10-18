# Nigerian Language Identifier — Evaluation Report

**Project:** Short-text identification for English, Hausa, Igbo and Yoruba  
**Status:** Preliminary evaluation; independent final validation pending  
**Model selected for application:** Character-level TF-IDF + Linear Support Vector Machine (LinearSVC)

## 1. Project objective
Develop a text classifier that predicts one of four language labels from a short user-entered phrase, and expose predictions through a Gradio interface.

## 2. Data and preparation
The working dataset contains **20,000 labelled text samples**, balanced at **5,000 per language**. The Nigerian-language examples were assembled from MasakhaNER 2.0 training material; English examples were drawn from the NLTK Brown Corpus. Data preparation included reconstructing sentences from tokens, filtering, removing exact duplicate text, sampling and shuffling. The working data was divided using a stratified 80/20 split: **16,000 training samples and 4,000 test samples** (1,000 test examples per language).

**Data caveat:** The sources differ in genre, formatting and typical sentence length. Consequently, the test score may partly reflect source-specific features. Exact-text deduplication was applied before splitting, but a systematic near-duplicate audit and independent source-matched benchmark have not been documented. Dataset licenses and redistribution rights should be verified before publishing a full copy.

## 3. Methods
Both models used Scikit-learn pipelines with character-level TF-IDF (2–5 character n-grams, `min_df=2`, `max_features=100000`). Two classifiers were evaluated:

- Logistic Regression (`max_iter=1000`, `random_state=42`)
- LinearSVC (`random_state=42`)

The LinearSVC model was selected provisionally based on the higher held-out test score. Repeated comparison on the same test data means that this is not an untouched final benchmark.

## 4. Quantitative results

| Metric | Logistic Regression | LinearSVC |
|---|---:|---:|
| Held-out test accuracy | 99.70% | **99.80%** |
| Correct out of 4,000 | 3,988 | **3,992** |
| Incorrect out of 4,000 | 12 | **8** |
| Diagnostic short-phrase accuracy | 87.50% | **87.50%** |
| Correct out of 16 diagnostic phrases | 14 | **14** |

The original Scikit-learn classification report for LinearSVC rounded precision, recall and F1 to two decimal places; all four class F1 values displayed as `1.00`. These rounded values should **not** be interpreted as perfect classification. For precise per-language results, export the unrounded `classification_report(..., output_dict=True)` and confusion matrix from the notebook.

## 5. Short-phrase diagnostic evaluation
Sixteen manually selected phrases (four per language) were tested. Both models correctly classified 14. For each model, the two incorrect predictions were:

| Text | Actual | Predicted |
|---|---|---|
| Na gode sosai | Hausa | English |
| Ina son ruwa | Hausa | English |

The other 14 diagnostic phrases were classified correctly. A substring check found no occurrences of four checked Hausa expressions in the balanced dataset; this does not rule out similar wording or shared terms. The diagnostic set is very small and has been inspected repeatedly during development; its score is **not** a generalizable real-world accuracy estimate.

For Logistic Regression, estimated class probabilities were:

| Phrase | English | Hausa | Igbo | Yoruba |
|---|---:|---:|---:|---:|
| Na gode sosai | 41.87% | 19.41% | 20.45% | 18.27% |
| Ina son ruwa | 35.69% | 31.73% | 14.32% | 18.26% |
| Sannu da zuwa | 12.58% | 68.42% | 7.90% | 11.10% |
| Yaya iyali? | 20.77% | 47.92% | 15.14% | 16.18% |

These are uncalibrated model probability estimates, not validated confidence guarantees.

## 6. Limitations and next validation steps
1. Evaluate on a **fresh**, independently sourced, human-verified short-phrase dataset with adequate examples per language. Do not use the already inspected 16 phrases to report final unbiased accuracy.
2. Include phrases without diacritics, informal spelling, short greetings, and code-switched text. The current model always selects one of the four labels, even when text is ambiguous or outside scope.
3. Review near-duplicates and dataset-source leakage; if possible, split by document or source rather than only by individual sentence.
4. Report unrounded per-class precision, recall and F1, plus a confusion matrix, on the independent benchmark.
5. Consider collecting more licensed, verified conversational Hausa text; compare changes on a development set before touching the final benchmark.
6. Test the saved model and Gradio application in a clean environment using the published repository files.

## 7. Conclusion
The prototype successfully trains, saves and serves a four-language text classifier. The selected LinearSVC achieved **99.80%** accuracy on the original held-out set, while both models achieved **87.50%** on a small short-phrase diagnostic. The observed Hausa-to-English errors demonstrate why real-world testing is necessary before deployment or broad accuracy claims.

**Evidence basis:** Figures and observations supplied from the user's Google Colab runs; no independent execution of the original notebook or model was performed when drafting this report.
