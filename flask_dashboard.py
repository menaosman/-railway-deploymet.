from flask import Flask, render_template_string, request, send_file, jsonify
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
        <script src='https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js'></script>
        <style>
            body { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background-color: #f8f9fa; }
            h2 { margin-bottom: 20px; }
            .btn-group { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h2>📊 Tweet Sentiment Analyzer</h2>
        <lottie-player src='https://assets2.lottiefiles.com/packages/lf20_puciaact.json' background='transparent' speed='1' style='width: 300px; height: 300px;' loop autoplay></lottie-player>
        <h2>📊 Tweet Sentiment Analyzer</h2>
        <lottie-player src='https://assets2.lottiefiles.com/packages/lf20_puciaact.json' background='transparent' speed='1' style='width: 300px; height: 300px;' loop autoplay></lottie-player>
        <div class='btn-group'>
            <a href='/dashboard' class='btn btn-primary'>📈 Dashboard (Visualizations)</a>
            <a href='/tweets_table' class='btn btn-dark'>📋 Tweets Table</a>
            <a href='/upload' class='btn btn-success'>📤 Upload CSV</a>
            <a href='/download_csv' class='btn btn-warning'>📥 Download CSV</a>
            <a href='/upload_mongo' class='btn btn-secondary'>📦 Upload to MongoDB</a>
            <a href='/fetch_mongo' class='btn btn-info'>📥 Fetch from MongoDB</a>
        </div>
        <div class='btn-group'>
            <a href='/dashboard' class='btn btn-primary'>📈 Dashboard</a>
            <a href='/upload' class='btn btn-success'>📤 Upload CSV</a>
            <a href='/download_csv' class='btn btn-warning'>📥 Download CSV</a>
            <a href='/upload_mongo' class='btn btn-secondary'>📦 Upload to MongoDB</a>
            <a href='/fetch_mongo' class='btn btn-info'>📥 Fetch from MongoDB</a>
        </div>
        
    </body>
    </html>
    """

@app.route('/dashboard', methods=['GET'])
def dashboard():
    keyword = request.args.get('keyword', '')
    data = list(collection.find({}, {"_id": 0, "Text": 1, "Sentiment": 1, "Timestamp": 1}))
    df = pd.DataFrame(data)

    if df.empty:
        return "<h3 class='text-center text-danger'>No Data Found in MongoDB!</h3>"

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
        <title>Dashboard</title>
        <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
    </head>
    <body class='container'>
        <h2 class='my-4'>📊 Sentiment Dashboard</h2>
        <form action='/dashboard' method='get' class='mb-4'>
            <input type='text' name='keyword' placeholder='Enter keyword' value='{keyword}' class='form-control' style='width: 300px; display: inline;'>
            <button type='submit' class='btn btn-info mt-2'>🔍 Filter</button>
        </form>
        <h3>📌 Sentiment Distribution</h3>
        <img src='data:image/png;base64,{sentiment_plot}' width='400'/>
        <h3>📈 Sentiment Over Time</h3>
        <img src='data:image/png;base64,{timeline_plot}' width='600'/>
        <h3>☁️ WordCloud</h3>
        <img src='data:image/png;base64,{wordcloud_plot}' width='600'/>
        <h3>📄 Tweets Table</h3>
        <a>🏠 Home</a>
    </body>
    </html>
    """
    return render_template_string(html)
@app.route('/tweets_table', methods=['GET'])
def tweets_table():
    data = list(collection.find({}, {"_id": 0, "Text": 1, "Sentiment": 1, "Timestamp": 1}))
    if not data:
        return "<h3 class='text-danger'>No Tweets Found!</h3><a href='/' class='btn btn-secondary'>🏠 Home</a>"

    df = pd.DataFrame(data)
    table_html = df.to_html(index=False, classes='table table-striped table-bordered')

    return f"""
    <h2>📋 Tweets Table</h2>
    {table_html}
    <a href='/' class='btn btn-secondary mt-4'>🏠 Home</a>
    """

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
            return "<h3 class='text-success'>✅ Upload Successful!</h3><a href='/dashboard' class='btn btn-primary'>Go to Dashboard</a>"

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload CSV</title>
        <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
    </head>
    <body class='container'>
        <h2 class='my-4'>📤 Upload CSV</h2>
        <form action='/upload' method='post' enctype='multipart/form-data'>
            <input type='file' name='file' class='form-control mb-3' style='width:300px;'>
            <button type='submit' class='btn btn-success'>Upload</button>
        </form>
        <a href='/' class='btn btn-secondary mt-4'>🏠 Home</a>
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
    @app.route('/upload_mongo', methods=['GET', 'POST'])
@app.route('/upload_mongo', methods=['GET', 'POST'])
def upload_mongo():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            df["BatchTimestamp"] = now
            records = df.to_dict("records")
            collection.insert_many(records)
            return "<h3 class='text-success'>✅ Upload to MongoDB Successful!</h3><a href='/' class='btn btn-primary'>🏠 Home</a>"
    return render_template_string("""
    <h2>📦 Upload Data to MongoDB</h2>
    <form action='/upload_mongo' method='post' enctype='multipart/form-data'>
        <input type='file' name='file' class='form-control mb-3' style='width:300px;'>
        <button type='submit' class='btn btn-success'>Upload</button>
    </form>
    <a href='/' class='btn btn-secondary mt-4'>🏠 Home</a>
    """)

@app.route('/fetch_mongo', methods=['GET'])
def fetch_mongo():
    data = list(collection.find({}, {"_id": 0, "Text": 1, "Sentiment": 1, "Timestamp": 1}))
    if not data:
        return "<h3 class='text-danger'>No data found in MongoDB!</h3><a href='/' class='btn btn-secondary'>🏠 Home</a>"
    df = pd.DataFrame(data)
    table_html = df.to_html(index=False, classes='table table-striped table-bordered')
    return f"""<h2>📥 Fetched Data from MongoDB</h2>{table_html}<a href='/' class='btn btn-secondary mt-4'>🏠 Home</a>"""

def encode_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
