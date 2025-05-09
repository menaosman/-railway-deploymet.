from flask import Flask, render_template_string, jsonify
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from wordcloud import WordCloud

app = Flask(__name__)

# MongoDB Config
mongo_uri = "mongodb+srv://biomedicalinformatics100:MyNewSecurePass%2123@cluster0.jilvfuv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
collection = client["sentiment_analysis"]["tweets"]

@app.route('/')
def home():
    return "<h2>Welcome to the Sentiment Dashboard! Try /dashboard</h2>"

@app.route('/dashboard')
def dashboard():
    data = list(collection.find({}, {"_id": 0, "Text": 1, "Sentiment": 1, "Timestamp": 1}))
    df = pd.DataFrame(data)

    if df.empty:
        return "<h3>No Data Found in MongoDB!</h3>"

    sentiment_plot = plot_sentiment_distribution(df)
    timeline_plot = plot_sentiment_over_time(df)
    wordcloud_plot = plot_wordcloud(df)

    html = f"""
    <h2>📊 Sentiment Dashboard</h2>
    <img src="data:image/png;base64,{sentiment_plot}" width="400"/>
    <h3>📈 Sentiment Over Time</h3>
    <img src="data:image/png;base64,{timeline_plot}" width="600"/>
    <h3>☁️ WordCloud</h3>
    <img src="data:image/png;base64,{wordcloud_plot}" width="600"/>
    """
    return render_template_string(html)

def plot_sentiment_distribution(df):
    plt.figure(figsize=(4, 4))
    sns.countplot(data=df, x="Sentiment", palette="Set2")
    return encode_plot()

def plot_sentiment_over_time(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    timeline = df.groupby([df["Timestamp"].dt.date, "Sentiment"]).size().unstack(fill_value=0)
    timeline.plot(kind='line', figsize=(6, 4))
    return encode_plot()

def plot_wordcloud(df):
    text = " ".join(df["Text"].dropna())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return encode_plot()

def encode_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
