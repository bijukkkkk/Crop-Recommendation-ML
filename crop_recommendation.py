
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Step 1: Create a folder to save results
os.makedirs("outputs", exist_ok=True)

# Step 2: Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

print("Dataset loaded successfully!")
print("Dataset size:", data.shape)
print(data.head())

# Step 3: Clean the dataset
data.columns = data.columns.str.strip()
data = data.drop_duplicates()

features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

data["label"] = data["label"].astype(str).str.strip()

for column in features:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna(subset=["label"])
data = data.dropna(subset=features)

X = data[features]
y = data["label"]

# Step 4: Show the number of each crop
plt.figure(figsize=(12, 6))
data["label"].value_counts().plot(kind="bar")
plt.title("Number of Samples for Each Crop")
plt.xlabel("Crop")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/crop_distribution.png")
plt.show()

# Step 5: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 6: Create the machine learning model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# Step 7: Train the model
print("Training the model...")
model.fit(X_train, y_train)
print("Training completed!")

# Step 8: Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, average="weighted", zero_division=0
)
recall = recall_score(
    y_test, predictions, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test, predictions, average="weighted", zero_division=0
)

print("\nMODEL RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Step 9: Save the results to a text file
with open("outputs/model_results.txt", "w") as file:
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n\n")
    file.write(classification_report(
        y_test, predictions, zero_division=0
    ))

# Step 10: Create a confusion matrix graph
cm = confusion_matrix(y_test, predictions, labels=model.classes_)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# Step 11: Show which features are important
classifier = model.named_steps["classifier"]

importance = pd.Series(
    classifier.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.plot(kind="bar", figsize=(9, 5))
plt.title("Importance of Each Feature")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.show()

# Step 12: Save the trained model
joblib.dump(model, "crop_model.joblib")
print("\nModel saved as crop_model.joblib")

# Step 13: Try a sample prediction
sample = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
}])

predicted_crop = model.predict(sample)[0]
print("\nSample predicted crop:", predicted_crop)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
