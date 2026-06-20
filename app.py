from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

model = joblib.load("savedmodel.pth")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        file = request.files["image"]

        img = Image.open(file)

        img = img.convert("L")

        img = img.resize((64, 64))

        img_array = np.array(img, dtype=np.float32)

        img_array = img_array / 255.0

        img_array = img_array.flatten().reshape(1, -1)

        prediction = model.predict(img_array)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )

