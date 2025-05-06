from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'sarcasm_model', 'model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'sarcasm_model', 'vectorizer.pkl'))


def clean_text(text):
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ''.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        headline = request.form['headline']
        if headline.strip() != "":
            cleaned_headline = clean_text(headline)
            headline_vector = vectorizer.transform([cleaned_headline])
            prediction = model.predict(headline_vector)[0]
            result = "Sarcastic" if prediction == 1 else "Not Sarcastic"
        else:
            result = "Please enter a valid headline."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
