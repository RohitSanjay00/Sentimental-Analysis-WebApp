from flask import Flask, render_template, request
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import tensorflow as tf
import numpy as np

loaded_model = tf.keras.models.load_model("Sentiment_model_1")
app = Flask(__name__, template_folder='templates')
cred = credentials.Certificate("sentimentclassification-113de-firebase-adminsdk-iv3fo-f1aab7aa58.json")  
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load BERT sentiment analysis model and tokenizer
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        complaint = request.form['complaint']
        model = request.form['model']
        if model == 'bert':
            process_bert(complaint)
        elif model == 'rohit':
            process_rohit(complaint)
        return "Complaint submitted successfully!"  # You can redirect or render another template here
    return render_template('index.html')

def process_bert(complaint):
    sentiment = predict_sentiment(complaint)
    save_to_firebase(complaint, sentiment)

def process_rohit(complaint):
    text = np.array([complaint])
    value = loaded_model.predict(text)
    value=labelling(value)
    label = mysentiment(value)
    save_to_firebase(complaint, label)


@app.route('/queue')
def queue():
    positive_complaints = fetch_complaints_from_firebase("Positive Sentiment")
    negative_complaints = fetch_complaints_from_firebase("Negative Sentiment")
    neutral_complaints = fetch_complaints_from_firebase("Neutral Sentiment")
    return render_template('queue.html', positive=positive_complaints, negative=negative_complaints, neutral=neutral_complaints)

def predict_sentiment(text):
    result = sentiment_classifier(text)[0]
    stars = result['label']
    if stars == '1 star':
        return 'NEGATIVE'
    elif stars == '5 stars':
        return 'POSITIVE'
    else:
        return 'NEUTRAL'

def save_to_firebase(text, sentiment):
    collection_name = "Positive Sentiment" if sentiment == 'POSITIVE' else ("Negative Sentiment" if sentiment == 'NEGATIVE' else "Neutral Sentiment")
    db.collection(collection_name).add({'text': text, 'sentiment': sentiment})

def fetch_complaints_from_firebase(collection_name):
    complaints = []
    docs = db.collection(collection_name).stream()
    for doc in docs:
        complaints.append(doc.to_dict())
    return complaints

def labelling(data):
    max_indices = np.argmax(data, axis=1)
    labels = np.zeros(data.shape[0], dtype=int)
    labels[max_indices == 0] = -1
    labels[max_indices == 1] = 0
    labels[max_indices == 2] = 1
    return labels[0]

def mysentiment(value):
    if value == -1:
        return 'NEGATIVE'
    elif value == 1: 
        return 'POSITIVE'
    else:
        return 'NEUTRAL'

if __name__ == '__main__':
    app.run(debug=True)