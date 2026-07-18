import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Core asset path initializations
MODEL_PATH = 'burnout_rf_model.pkl'
SCALER_PATH = 'app_scaler.pkl'
ENCODER_PATH = 'app_encoder.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    print("🧠 ML Engine Model & TargetEncoder Components loaded successfully.")
else:
    print("⚠️ Warning: Production pipeline artifacts (.pkl files) are missing from root folder!")
    model, scaler, encoder = None, None, None

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler or not encoder:
        return jsonify({'status': 'error', 'message': 'Pipeline assets uninitialized on host server.'}), 500
        
    try:
        data = request.get_json()
        
        # 1. Reconstruct raw single-row DataFrame matching training layout EXACTLY
        input_df = pd.DataFrame([{
            'Major_Category': data['Major_Category'],
            'Year_of_Study': data['Year_of_Study'],
            'Pre_Semester_GPA': float(data['Pre_Semester_GPA']),
            'Weekly_GenAI_Hours': float(data['Weekly_GenAI_Hours']),
            'Primary_Use_Case': data['Primary_Use_Case'],
            'Prompt_Engineering_Skill': data['Prompt_Engineering_Skill'],
            'Tool_Diversity': int(data['Tool_Diversity']),
            'Paid_Subscription': data['Paid_Subscription'],  # Kept raw to match boolean check
            'Traditional_Study_Hours': float(data['Traditional_Study_Hours']),
            'Perceived_AI_Dependency': int(data['Perceived_AI_Dependency']),
            'Institutional_Policy': data['Institutional_Policy'],
            'Anxiety_Level_During_Exams': int(data['Anxiety_Level_During_Exams']),
            'Skill_Retention_Score': float(data['Skill_Retention_Score'])
        }])
        
        # 2. Step from training line 26: Convert boolean field values into integers (0 or 1)
        input_df['Paid_Subscription'] = (input_df['Paid_Subscription'].astype(str).str.lower() == 'true').astype(int)
        
        # 3. Ensure columns match the training order perfectly before encoding
        feature_order = [
            'Major_Category', 'Year_of_Study', 'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 
            'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Tool_Diversity', 
            'Paid_Subscription', 'Traditional_Study_Hours', 'Perceived_AI_Dependency', 
            'Institutional_Policy', 'Anxiety_Level_During_Exams', 'Skill_Retention_Score'
        ]
        input_df = input_df[feature_order]

        # 4. Transform categorical variables using your TargetEncoder asset (keeps shape at 1 row, 13 columns)
        X_encoded = encoder.transform(input_df)
        
        # 5. Scale all columns together (exactly as done in training line 36)
        X_scaled = scaler.transform(X_encoded)
        
        # 6. Model classification execution
        prediction_id = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = max(probabilities) * 100
        
        # Map training numbers back to string labels for your UI dashboard frontend
        reverse_target_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        burnout_label = reverse_target_map.get(prediction_id, 'Medium')
        
        return jsonify({
            'status': 'success',
            'burnout_risk': burnout_label,
            'confidence_score': f"{confidence:.1f}%"
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Logs clean trace data onto command prompt window
        return jsonify({'status': 'error', 'message': f"Pipeline Execution Fault: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)