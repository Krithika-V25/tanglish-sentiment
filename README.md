# Tanglish Sentiment Analyzer

Classifies Tamil-English (Tanglish) social media comments as positive, negative, or neutral.

## Approach
- Model: character n-gram TF-IDF + Logistic Regression (scikit-learn)
- Data: custom-written Tanglish comments modelled on social media text (seed set of 40, expanding)
- Why character n-grams: Tanglish has no standard spelling (semma / sema / semmaa)

## Run
pip install pandas scikit-learn joblib
python train.py

## Results
Baseline on a small seed dataset. Scores are rough because the test set is only 8 comments.

## Next steps
- Expand the dataset and retrain
- Evaluate on the public Dravidian CodeMix dataset
- Compare against an LLM on the same test comments
- Streamlit demo UI