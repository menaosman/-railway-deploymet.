from flask import Flask, render_template
from pymongo import MongoClient
import pandas as pd
import plotly.express as px

app = Flask(__name__)

# MongoDB Setup
mongo_uri = "mongodb+srv://biomedicalinformatics100:MyNewSecurePass%2123@cluster0.jilvfuv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
collection = client["sentiment_analysis"]["tweets"]

@app.route("/")
def home():
    data = list(collection.find({}, {"_id": 0, "Text": 1, "Sentiment": 1, "Timestamp": 1}))
    df = pd.DataFrame(data)

    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    fig = px.pie(sentiment_counts, values='Sentiment', names='index', title='Sentiment Distribution')

    chart_html = fig.to_html(full_html=False)

    return render_template("dashboard.html", chart_html=chart_html)

if __name__ == "__main__":
    app.run(debug=True)
