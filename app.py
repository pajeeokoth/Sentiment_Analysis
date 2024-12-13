from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
import pickle
import spacy
from flask import Flask, request, jsonify

# read the vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# load the pretrained model
model = pickle.load(open('logistic_model.pkl', 'rb'))
# model = load_model('best_model1.h5')

# load the word embeddingsclear
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

@app.route('/predict_sentiment', methods=["POST"])
def classify_tweet():
    # get the tweet
    data = request.json
    tweet = data["text"]
    # clean up the tweet
    tweet = str(tweet).lower().replace('#', '').replace('@', '').replace(r'http[s]?://\S+|www.\S+', '')
    # get document term vector
    message_vect = vectorizer.transform([tweet])
    #perform sentiment prediction
    prediction = model.predict(message_vect)[0]
    return jsonify({'Positive' if prediction == 1 else 'Negative'})

# def predict_sentiment(t):
#     data = request.json
#     tweet = data['text']
#     max_len = 85 #40
#     sequence = tokenizer.texts_to_sequences([tweet])
#     text = pad_sequences(sequence, maxlen=max_len)
#     # sent = sentiment[np.around(model.predict(text), decimals=0).argmax(axis=1)[0]]
#     prediction = (model.predict(text) > 0.5).astype(int)
#     return 'Positive' if prediction == 1 else 'Negative'

if __name__ == '__main__':
    app.run(port=5000)