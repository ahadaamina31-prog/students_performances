import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("student_performance.csv")

# Features (Input)
X = data[["StudyHours", "Attendance", "PreviousScore"]]

# Target (Output)
y = data["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict a new student
new_student = [[6, 85, 70]]

result = model.predict(new_student)

if result[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")