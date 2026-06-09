from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Loading Olivetti Faces Dataset...")

faces = fetch_olivetti_faces()

X = faces.data
y = faces.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("Loading saved model...")

model = joblib.load("savedmodel.pth")

print("Evaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {accuracy * 100:.2f}%")