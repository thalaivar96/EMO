from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = Flask(__name__)

model = DistilBertForSequenceClassification.from_pretrained("emotion_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("emotion_model")
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    prediction = labels[outputs.logits.argmax(-1).item()]
    return jsonify({"emotion": prediction})

if __name__ == "__main__":
    app.run(debug=True)
