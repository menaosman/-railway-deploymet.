from flask import Flask, render_template_string, request, send_file
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from wordcloud import WordCloud
import os
from datetime import datetime

app = Flask(__name__)

# MongoDB Config
mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://biomedicalinformatics100:MyNewSecurePass@cluster0.abcd123.mongodb.net/?retryWrites=true&w=majority")
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
collection = client["sentiment_analysis"]["tweets"]

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tweet Sentiment Analyzer</title>
        <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
        <style>
            body { padding: 40px; background-color: #f8f9fa; }
            h2 { margin-bottom: 20px; }
            button { margin-right: 10px; }
        </style>
    </head>
    <body>
        <h2>📊 Tweet Sentiment Analyzer</h2>
        <a href='/dashboard'><button class='btn btn-primary'>📈 Go to Dashboard</button></a>
        <a href='/upload'><button class='btn btn-success'>📤 Upload CSV</button></a>
    </body>
    </html>
    """

@app.route('/dashboard', methods=['GET'])
def dashboard():
    keyword = request.args.get('keyword', '')
    data = list(collection.find({}, {"_id": 0, "Text": 1, "Sentiment": 1, "Timestamp": 1}))
    df = pd.DataFrame(data)

    if df.empty:
        return "<h3>No Data Found in MongoDB!</h3>"

    if keyword:
        df = df[df["Text"].str.contains(keyword, case=False)]

    sentiment_plot = plot_sentiment_distribution(df)
    timeline_plot = plot_sentiment_over_time(df)
    wordcloud_plot = plot_wordcloud(df)
    table_html = df.to_html(index=False, classes='table table-striped table-bordered')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Dashboard</title>
        <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
        <style>
            body {{ padding: 20px; background-color: #f8f9fa; }}
            img {{ margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h2>📊 Sentiment Dashboard</h2>
        <form action='/dashboard' method='get' class='mb-4'>
            <input type='text' name='keyword' placeholder='Enter keyword' value='{keyword}' class='form-control' style='width: 300px; display: inline;'>
            <button type='submit' class='btn btn-info'>🔍 Filter</button>
        </form>
        <h3>📌 Sentiment Distribution</h3>
        <img src='data:image/png;base64,{sentiment_plot}' width='400'/>
        <h3>📈 Sentiment Over Time</h3>
        <img src='data:image/png;base64,{timeline_plot}' width='600'/>
        <h3>☁️ WordCloud</h3>
        <img src='data:image/png;base64,{wordcloud_plot}' width='600'/>
        <h3>📄 Raw Tweets Table</h3>
        {table_html}
        <br><br>
        <a href='/download_csv'><button class='btn btn-warning'>📥 Download CSV</button></a>
        <a href='/'><button class='btn btn-secondary'>🏠 Home</button></a>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            df["BatchTimestamp"] = now
            records = df.to_dict("records")
            collection.insert_many(records)
            return "<h3>✅ Upload Successful!</h3><a href='/dashboard'>Go to Dashboard</a>"

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload CSV</title>
        <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
        <style> body {{ padding: 40px; background-color: #f8f9fa; }} </style>
    </head>
    <body>
        <h2>📤 Upload CSV</h2>
        <form action='/upload' method='post' enctype='multipart/form-data'>
            <input type='file' name='file' class='form-control' style='width:300px;'>
            <button type='submit' class='btn btn-success mt-2'>Upload</button>
        </form>
        <br><a href='/'><button class='btn btn-secondary'>🏠 Home</button></a>
    </body>
    </html>
    """

@app.route('/download_csv')
def download_csv():
    data = list(collection.find({}, {"_id": 0, "Text": 1, "Sentiment": 1, "Timestamp": 1}))
    df = pd.DataFrame(data)
    csv_io = io.StringIO()
    df.to_csv(csv_io, index=False)
    csv_io.seek(0)
    return send_file(io.BytesIO(csv_io.getvalue().encode()),
                     mimetype='text/csv',
                     download_name="sentiment_data.csv",
                     as_attachment=True)

def plot_sentiment_distribution(df):
    plt.figure(figsize=(4, 4))
    sns.countplot(data=df, x="Sentiment", palette="Set2")
    plt.title("Sentiment Distribution")
    return encode_plot()

def plot_sentiment_over_time(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    timeline = df.groupby([df["Timestamp"].dt.date, "Sentiment"]).size().unstack(fill_value=0)
    timeline.plot(kind='line', figsize=(6, 4))
    plt.title("Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Tweet Count")
    return encode_plot()

def plot_wordcloud(df):
    text = " ".join(df["Text"].dropna())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return encode_plot()

def encode_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
