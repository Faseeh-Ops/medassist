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