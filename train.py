from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

print("Loading Olivetti Faces Dataset...")

# Load dataset
faces = fetch_olivetti_faces()

X = faces.data
y = faces.target

print("Splitting dataset into 70% train and 30% test...")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("Training Decision Tree Classifier...")

# Train model
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("Saving model as savedmodel.pth...")

# Save model
joblib.dump(model, "savedmodel.pth")

print("Model saved successfully")