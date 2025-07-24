# MedAssist - Patient Chat → Department + Wait Time Estimator

This NLP-powered application predicts the relevant medical department and estimated wait time based on a patient's written symptom description.

## Tasks:
- 🩺 Department Classification (Cardiology, Ortho, etc.)
- ⏳ Wait Time Regression (in minutes)

## Tech Stack:
- Python (NLP + ML)
- Scikit-learn
- Tkinter for GUI
- Trained on Symptom2Disease dataset (Kaggle)

## How to Use:
1. Place the dataset in `data/symptom2disease.csv`
2. Run `prepare_data.py` to generate `prepared.csv`
3. Run `train.py` to train models and save `.pkl` files
4. Run `app.py` to launch the GUI

   <img width="3825" height="1644" alt="image" src="https://github.com/user-attachments/assets/76db6f8e-8009-4bc3-8ba7-b3a8a51f8315" />

   <img width="3840" height="1644" alt="image" src="https://github.com/user-attachments/assets/a0c9c762-bee0-4687-b413-8ef0d4ffaf69" />

   <img width="1185" height="993" alt="image" src="https://github.com/user-attachments/assets/e62e0a6e-dff4-40ac-be70-d450daf61d23" />


