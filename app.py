from flask import Flask, jsonify, render_template, request

from src.classifier import CommentClassifier

app = Flask(__name__)

clf = CommentClassifier()
if clf.is_trained():
    clf.load()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/report")
def report():
    return render_template("report.html")


@app.route("/report/setfit")
def report_setfit():
    return render_template("report_setfit.html")


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    comment = (data or {}).get("comment", "").strip()
    if not comment:
        return jsonify({"error": "No comment provided"}), 400
    if not clf.is_trained():
        return jsonify({"error": "Model not trained. Run: python main.py train"}), 503
    results = clf.predict(comment)
    return jsonify({"comment": comment, "labels": results})
