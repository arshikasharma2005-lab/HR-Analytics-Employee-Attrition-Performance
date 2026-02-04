import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())

# Target Variable Distribution
print("\nAttrition Count:\n", df["Attrition"].value_counts())

plt.figure()
sns.countplot(x="Attrition", data=df)
plt.title("Attrition Count")
plt.show()

plt.figure()
sns.countplot(x="Department", hue="Attrition", data=df)
plt.title("Attrition vs Department")
plt.xticks(rotation=30)
plt.show()

plt.figure()
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure()
sns.countplot(x="JobSatisfaction", hue="Attrition", data=df)
plt.title("Job Satisfaction vs Attrition")
plt.show()



df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})


le = LabelEncoder()
for column in df.select_dtypes(include=["object"]).columns:
    df[column] = le.fit_transform(df[column])


X = df.drop("Attrition", axis=1)
y = df["Attrition"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nModel Used: Random Forest Classifier")
print("The model predicts whether an employee is likely to leave the company.")
print("Further improvement can be done using feature selection and hyperparameter tuning.")
