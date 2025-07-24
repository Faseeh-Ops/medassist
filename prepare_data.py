import pandas as pd
import os
# Load dataset
df = pd.read_csv("data/symptom2disease.csv")

mapping = {
    "Psoriasis": ("Dermatology", 15),
    "Asthma": ("Pulmonology", 25),
    "Stroke": ("Neurology", 35),
    "Osteoarthritis": ("Orthopedics", 40),
    "Heart Attack": ("Cardiology", 30),
    "Gastritis": ("Gastroenterology", 30),
    "Migraine": ("Neurology", 20),
    "Diabetes": ("Endocrinology", 25),

}

def map_department(disease):
    return mapping.get(disease, ("General Medicine", 30))

df[["department", "wait_time"]] = df["label"].apply(lambda d: pd.Series(map_department(d)))

os.makedirs("data", exist_ok=True)
df[["text", "department", "wait_time"]].to_csv("data/prepared.csv", index=False)
print(" Data prepared and saved to data/prepared.csv")
