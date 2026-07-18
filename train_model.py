import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load Dataset
df = pd.read_csv('ai_student_impact_dataset_20000_rows.csv')

# 2. Separate Features and Target
feature_cols = [
    'Major_Category', 'Year_of_Study', 'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 
    'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Tool_Diversity', 
    'Paid_Subscription', 'Traditional_Study_Hours', 'Perceived_AI_Dependency', 
    'Institutional_Policy', 'Anxiety_Level_During_Exams', 'Skill_Retention_Score'
]
target_col = 'Burnout_Risk_Level'

X = df[feature_cols].copy()

# Manually map string labels to numbers to avoid encoder conflicts
target_map = {'Low': 0, 'Medium': 1, 'High': 2}
y = df[target_col].map(target_map).copy()

# Convert Boolean column to integer
X['Paid_Subscription'] = X['Paid_Subscription'].astype(int)

# 3. Handle Categorical Encoding using Target Encoding
categorical_cols = ['Major_Category', 'Year_of_Study', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Institutional_Policy']
encoder = ce.TargetEncoder(cols=categorical_cols)
X_encoded = encoder.fit_transform(X, y)

# 4. Train-Test Split and Scaling
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"🎯 Production Model Trained Successfully!")
print(f"Validation Accuracy: {model.score(X_test_scaled, y_test) * 100:.2f}%")

# 6. Save Artifacts for Flask App deployment
with open('burnout_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('app_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('app_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)