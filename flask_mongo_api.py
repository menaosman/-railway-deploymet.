from flask import Flask, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import os
@app.route('/')
def home():
    return "<h1>Welcome to the Flask API Dashboard!</h1><p>Use <a href='/fetch'>/fetch</a> to get tweets data.</p>"

app = Flask(__name__)
CORS(app)

# MongoDB connection using Environment Variable
mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://biomedicalinformatics100:MyNewSecurePass%2123@cluster0.jilvfuv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
collection = client["sentiment_analysis"]["tweets"]

@app.route('/fetch', methods=['GET'])
def fetch_tweets():
    try:
        cursor = collection.find({}, {"_id": 0, "Text": 1, "Sentiment": 1, "Timestamp": 1})
        tweets = list(cursor)

        for tweet in tweets:
            ts = tweet.get("Timestamp")
            if isinstance(ts, datetime):
                tweet["Timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                tweet["Timestamp"] = "N/A"

        return jsonify(tweets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
