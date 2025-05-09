from flask import Flask, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import io
import base64

app = Flask(__name__)

# MongoDB Connection
mongo_uri = "mongodb+srv://biomedicalinformatics100:MyNewSecurePass%2123@cluster0.jilvfuv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
collection = client["sentiment_analysis"]["tweets"]

@app.route('/')
def home():
    return "<h1>Welcome to the Flask Dashboard!</h1><p>Use /dashboard to view analytics.</p>"

@app.route('/dashboard')
def dashboard():
    data = list(collection.find({}, {"_id": 0, "Text": 1, "Sentiment": 1, "Timestamp": 1}))
    df = pd.DataFrame(data)

    # Simple Sentiment Plot
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="Sentiment")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    html = f"""
    <h1>Sentiment Distribution</h1>
    <img src="data:image/png;base64,{plot_url}">
    """
    return html

if __name__ == '__main__':
    app.run(debug=True)
