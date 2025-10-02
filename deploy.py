import gradio as gr
import pickle
import numpy as np

# Load the saved model, scaler, and encoder
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))  # One encoder for both Geography and Gender

# Function to safely encode values
def safe_encode(encoder, value, fallback_value=-1):
    try:
        # LabelEncoder returns a 1D array, so we just need [0] instead of [0, 0]
        return encoder.transform([value])[0]
    except ValueError:
        return fallback_value

#  Function to make predictions
def predict_churn(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    input_data = [[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]
    
    # Encode categorical variables and scale input data safely
    input_data[0][1] = safe_encode(label_encoder, Geography)  # Encode Geography
    input_data[0][2] = safe_encode(label_encoder, Gender)  # Encode Gender
    
    input_data = scaler.transform(input_data)
    
    # Predict using the model
    prediction = model.predict(input_data)
    return "Customer most likely to exit" if prediction[0] == 1 else "Customer most likely to stay"

# Define the Gradio interface
inputs = [
    gr.components.Number(label="CreditScore"),
    gr.components.Textbox(label="Geography"),
    gr.components.Textbox(label="Gender"),
    gr.components.Number(label="Age"),
    gr.components.Number(label="Tenure"),
    gr.components.Number(label="Balance"),
    gr.components.Number(label="NumOfProducts"),
    gr.components.Checkbox(label="HasCrCard"),
    gr.components.Checkbox(label="IsActiveMember"),
    gr.components.Number(label="EstimatedSalary")
]

outputs = gr.components.Textbox(label="Prediction")

gr.Interface(fn=predict_churn, inputs=inputs, outputs=outputs, title="Bank Customer Churn Prediction").launch(share=True)
