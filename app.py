import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load and prepare the dataset
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
df = pd.read_csv(url)

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. Train the Logistic Regression model
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)

# 3. Prediction function for Gradio UI
def predict_heart_disease(age, sex_str, cp_str, trestbps, chol, fbs_str, restecg_str, thalach, exang_str, oldpeak, slope_str, ca, thal_str):
    sex = 1 if sex_str == "Male" else 0
    cp_map = {"Typical Angina (0)": 0, "Atypical Angina (1)": 1, "Non-anginal Pain (2)": 2, "Asymptomatic (3)": 3}
    fbs = 1 if fbs_str == "Yes (> 120 mg/dl)" else 0
    restecg_map = {"Normal (0)": 0, "ST-T Wave Abnormality (1)": 1, "Left Ventricular Hypertrophy (2)": 2}
    exang = 1 if exang_str == "Yes" else 0
    slope_map = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
    thal_map = {"Normal (1)": 1, "Fixed Defect (2)": 2, "Reversible Defect (3)": 3}

    patient_data = pd.DataFrame([[
        age, sex, cp_map[cp_str], trestbps, chol, fbs,
        restecg_map[restecg_str], thalach, exang, oldpeak,
        slope_map[slope_str], ca, thal_map[thal_str]
    ]], columns=X.columns)

    patient_scaled = scaler.transform(patient_data)
    prediction = log_model.predict(patient_scaled).item()
    healthy_prob, disease_prob = (log_model.predict_proba(patient_scaled) * 100).flatten()

    if prediction == 1:
        return (
            f"HIGH RISK DETECTED ({disease_prob:.1f}% Probability)\n\n"
            f"• Heart Disease Risk: {disease_prob:.1f}%\n"
            f"• Healthy Likelihood: {healthy_prob:.1f}%\n"
            f"• Recommendation: Consult a cardiologist for further clinical evaluation."
        )
    else:
        return (
            f"LOW RISK / HEALTHY ({healthy_prob:.1f}% Confidence)\n\n"
            f"• Healthy Likelihood: {healthy_prob:.1f}%\n"
            f"• Heart Disease Risk: {disease_prob:.1f}%\n"
            f"• Recommendation: Maintain a healthy lifestyle and regular checkups."
        )

# 4. Build the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as app:
    gr.Markdown("# ❤️ Smart Heart Disease Prediction System\n**AI-Powered Clinical Risk Assessment Tool**")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 👤 Basic Information")
                with gr.Row():
                    age = gr.Slider(20, 100, value=58, step=1, label="Age (Years)")
                    sex = gr.Radio(["Male", "Female"], value="Male", label="Biological Sex")

            with gr.Group():
                gr.Markdown("### 🩺 Vitals & Blood Tests")
                with gr.Row():
                    trestbps = gr.Number(value=150, label="Resting Blood Pressure (mm Hg)")
                    chol = gr.Number(value=270, label="Serum Cholesterol (mg/dl)")
                with gr.Row():
                    thalach = gr.Slider(60, 220, value=111, step=1, label="Max Heart Rate Achieved")
                    fbs = gr.Radio(["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"], value="No (<= 120 mg/dl)", label="Fasting Blood Sugar")

            with gr.Accordion("🔬 Advanced Cardiology Metrics", open=True):
                with gr.Row():
                    cp = gr.Dropdown(["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"], value="Typical Angina (0)", label="Chest Pain Type")
                    exang = gr.Radio(["No", "Yes"], value="Yes", label="Exercise Induced Angina")
                with gr.Row():
                    restecg = gr.Dropdown(["Normal (0)", "ST-T Wave Abnormality (1)", "Left Ventricular Hypertrophy (2)"], value="Normal (0)", label="Resting ECG Results")
                    oldpeak = gr.Slider(0.0, 6.5, value=0.8, step=0.1, label="ST Depression (Oldpeak)")
                with gr.Row():
                    slope = gr.Dropdown(["Upsloping (0)", "Flat (1)", "Downsloping (2)"], value="Downsloping (2)", label="ST Segment Slope")
                    ca = gr.Slider(0, 3, value=0, step=1, label="Major Vessels Colored (0-3)")
                    thal = gr.Dropdown(["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"], value="Reversible Defect (3)", label="Thalassemia Status")

            with gr.Row():
                clear_btn = gr.ClearButton(value="🔄 Reset Form")
                submit_btn = gr.Button("🔍 Analyze Patient Risk", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📋 Diagnostic Report")
            output_box = gr.Textbox(label="AI Assessment Summary", lines=8)

    gr.Examples(
        examples=[
            [58, "Male", "Typical Angina (0)", 150, 270, "No (<= 120 mg/dl)", "Normal (0)", 111, "Yes", 0.8, "Downsloping (2)", 0, "Reversible Defect (3)"],
            [42, "Female", "Non-anginal Pain (2)", 120, 204, "No (<= 120 mg/dl)", "Normal (0)", 172, "No", 0.0, "Upsloping (0)", 0, "Fixed Defect (2)"]
        ],
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        label="⚡ Quick Test Cases"
    )

    all_inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    clear_btn.add(all_inputs + [output_box])
    submit_btn.click(fn=predict_heart_disease, inputs=all_inputs, outputs=output_box)

if __name__ == "__main__":
    app.launch(share=True)
