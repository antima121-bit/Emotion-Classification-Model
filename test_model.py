import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('text_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Test sentences
test_sentences = [
    "I am very happy",
    "I am very angry",
    "I am sad",
    "I am scared",
    "I am surprised",
    "This is normal"
]

# Print model classes
print("Model classes:", model.classes_)

# Test each sentence
for sentence in test_sentences:
    # Transform the text
    text_vectorized = vectorizer.transform([sentence])
    # Get prediction
    prediction = model.predict(text_vectorized)
    # Get probabilities
    probabilities = model.predict_proba(text_vectorized)[0]
    
    print(f"\nInput: {sentence}")
    print(f"Predicted class: {prediction[0]}")
    print(f"Probabilities: {probabilities}")
    print(f"Most likely emotion: {model.classes_[prediction[0]]}")
    print(f"Confidence: {probabilities[prediction[0]]*100:.2f}%") 
