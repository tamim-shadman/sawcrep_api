import joblib
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from flaskcors import CORS

app = Flask(__name__)
CORS(app)
# Load the model
model = joblib.load('NB_model.joblib')

# Load the CountVectorizer
vectorizer = joblib.load('CountVectorizer_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get review text from request
        review_text = request.json['review_text']

        # Transform the review text using the loaded CountVectorizer
        review_vector = vectorizer.transform([review_text])

        # Make prediction
        predicted_recommendation = model.predict(review_vector)

        # Return prediction as JSON (assuming predicted_recommendation is an array)
        return jsonify({'recommended_ind': int(predicted_recommendation[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
