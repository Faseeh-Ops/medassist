import gradio as gr
import joblib

# Load models
clf = joblib.load("models/department_classifier.pkl")
reg = joblib.load("models/wait_time_predictor.pkl")
le = joblib.load("models/label_encoder.pkl")


# Prediction function
def predict_medassist(text):
    if not text.strip():
        return "Please enter symptom description.", ""

    dept_code = clf.predict([text])[0]
    wait_time = reg.predict([text])[0]
    department = le.inverse_transform([dept_code])[0]

    return department, f"{int(wait_time)} minutes"


# Gradio Interface
interface = gr.Interface(
    fn=predict_medassist,
    inputs=gr.Textbox(lines=4, label="Describe your symptoms"),
    outputs=[
        gr.Textbox(label="Predicted Department"),
        gr.Textbox(label="Estimated Wait Time")
    ],
    title="🩺 MedAssist - Patient Triage Assistant",
    description="Enter patient symptoms below to get predicted department and wait time."
)

if __name__ == "__main__":
    interface.launch()
