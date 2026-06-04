import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_excel("Student_Performence.xlsx")

# Create target column (PASS if math score >= 50)
data["pass_fail"] = (data["math score"] >= 50).astype(int)

# Categorical columns
categorical_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

# Store encoders (IMPORTANT)
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and target
X = data.drop(["math score", "pass_fail"], axis=1)
y = data["pass_fail"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model + encoders (VERY IMPORTANT)
joblib.dump(model, "student_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model + encoders saved successfully")