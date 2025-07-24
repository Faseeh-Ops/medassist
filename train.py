import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

df = pd.read_csv("data/prepared.csv")
le = LabelEncoder()
df["dept_encoded"] = le.fit_transform(df["department"])

X = df["text"]

y_clf = df["dept_encoded"]
y_reg = df["wait_time"]

X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42
)

clf = Pipeline([("tfidf", TfidfVectorizer()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
clf.fit(X_train, y_clf_train)

reg = Pipeline([("tfidf", TfidfVectorizer()), ("model", RandomForestRegressor(n_estimators=100, random_state=42))])
reg.fit(X_train, y_reg_train)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/department_classifier.pkl")
joblib.dump(reg, "models/wait_time_predictor.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Models trained and saved in 'models/'")