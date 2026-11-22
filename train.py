import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

df = pd.read_csv("data.csv").dropna()
df["text"] = df["text"].str.lower()

print(df["label"].value_counts())  # check class imbalance

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds, zero_division=0))

joblib.dump(model, "model.pkl")

# Look at mistakes: this is where you learn the most
errors = X_test[preds != y_test]
print(errors.head(20))