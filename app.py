import tkinter as tk
from tkinter import messagebox
import joblib

clf = joblib.load("models/department_classifier.pkl")
reg = joblib.load("models/wait_time_predictor.pkl")
le = joblib.load("models/label_encoder.pkl")

def predict():
    text = input_field.get("1.0", "end").strip()
    if not text:
        messagebox.showwarning("Input Needed", "Please describe your symptoms.")
        return

    dept_code = clf.predict([text])[0]
    wait = reg.predict([text])[0]
    dept = le.inverse_transform([dept_code])[0]

    result_label.config(
        text=f"🩺 Department: {dept}\n⏱️ Estimated Wait Time: {int(wait)} minutes"
    )

root = tk.Tk()
root.title("MedAssist - NLP Triage")
root.geometry("400x300")
root.config(bg="#f8f9fa")

tk.Label(root, text="Enter patient symptoms:", font=("Helvetica", 12, "bold"), bg="#f8f9fa").pack(pady=10)
input_field = tk.Text(root, height=5, width=40)
input_field.pack(pady=5)
tk.Button(root, text="Predict", command=predict, bg="#007bff", fg="white", font=("Helvetica", 11)).pack(pady=10)
result_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#f8f9fa")
result_label.pack(pady=20)
root.mainloop()