# Nigerian Language Identifier

A beginner-friendly machine learning capstone project that predicts whether a short text is written in **Yoruba, Hausa, Igbo, or English**.

## Project problem
Applications that serve multilingual Nigerian users can benefit from identifying the language of incoming text. This prototype accepts a text phrase and returns one of four language labels. It does **not** translate text or reliably handle code-switching.

## Dataset
- 20,000 samples in the balanced development dataset: 5,000 each for Yoruba, Hausa, Igbo, and English.
- Nigerian-language text was reconstructed from the MasakhaNER 2.0 named-entity recognition corpus; English sentences came from the NLTK Brown Corpus.
- Samples were cleaned, deduplicated by exact text, balanced, and split into 16,000 training and 4,000 test examples with stratification.
- **Important:** These sources differ in genre and formatting. High in-dataset accuracy may not generalize to informal messages. Verify source licensing and attribution before redistributing any dataset files.

## Approach
1. Convert text to character-level TF-IDF features (2–5 character n-grams).
2. Train and compare Logistic Regression and Linear Support Vector Machine (LinearSVC) classifiers.
3. Evaluate on a stratified 20% held-out split and a small set of conversational phrases.
4. Serve the selected LinearSVC pipeline using Gradio.

## Results

| Evaluation | Logistic Regression | Linear SVM |
| --- | ---: | ---: |
| Original held-out test set (4,000 samples) | 99.70% | 99.80% |
| Short-phrase diagnostic (16 samples) | 87.50% | 87.50% |

The SVM correctly classified 3,992 of 4,000 original test examples. Both models misclassified **“Na gode sosai”** and **“Ina son ruwa”** (Hausa) as English in the 16-phrase diagnostic. The short-phrase test is small and has already been used for model comparison, so it is **not an independent estimate** of deployment performance. The original test set has also been used to compare models; a fresh external evaluation is recommended.

## Run locally

Install Python, then from this repository folder:

```bash
python -m pip install -r requirements.txt
python app.py
```

Place your own trained model file named `nigerian_language_identifier_svm.pkl` in the same folder as `app.py` before running. **Only load trusted `.pkl` files:** pickle-based formats can execute code when loaded. The model was trained with scikit-learn; using the same library version as the training environment is recommended.

## Files
- `app.py`: Gradio user interface.
- `requirements.txt`: application dependencies.
- `Nigerian_Language_Identifier.ipynb`: data preparation, training, evaluation, and demonstration (upload your completed Colab notebook).
- `nigerian_language_identifier_svm.pkl`: trained model (upload your trusted file if GitHub size limits permit).

## Limitations and next steps
- Evaluate on a larger, independently collected and verified set of everyday phrases.
- Test short messages without diacritics, slang, and mixed-language input.
- Investigate Hausa errors using additional licensed conversational data.
- Audit near-duplicate sentences and possible corpus-source artifacts.
- Add an “uncertain / unsupported language” response after suitable calibration and validation.

## Acknowledgements
MasakhaNER 2.0 / Masakhane and the NLTK Brown Corpus. Check and retain their applicable licences and attribution requirements before redistributing source data.
