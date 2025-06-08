import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Set page config
st.set_page_config(
    page_title="Emotion Classification",
    page_icon="😊",
    layout="wide"
)

def preprocess_text(text):
    """Preprocess the input text for better emotion detection"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def get_emotion_keywords():
    """Return a dictionary of emotion keywords for validation"""
    return {
        'joy': ['happy', 'joy', 'delighted', 'pleased', 'glad', 'cheerful', 'excited', 'thrilled'],
        'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'gloomy', 'down', 'blue', 'sorrow', 'not happy', 'not my day', 'bad day', 'not good'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'enraged', 'frustrated'],
        'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous'],
        'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'unexpected'],
        'neutral': ['okay', 'fine', 'alright', 'normal', 'usual', 'regular', 'standard']
    }

def validate_emotion(text, predicted_emotion, confidence):
    """Validate the predicted emotion against keywords in the text"""
    emotion_keywords = get_emotion_keywords()
    text_lower = text.lower()
    
    # Check for negative emotions first
    if any(keyword in text_lower for keyword in emotion_keywords['sadness']):
        return 'sadness'
    if any(keyword in text_lower for keyword in emotion_keywords['anger']):
        return 'anger'
    
    # Then check other emotions
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            if emotion != predicted_emotion and confidence < 0.7:  # Lowered threshold
                return emotion
    return predicted_emotion

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('text_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Load model and vectorizer
try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Title and description
st.title('😊 Emotion Classification')
st.write('Enter text to classify its emotion using our trained model.')

# Text input
text_input = st.text_area('Input Text:', height=100, 
                         placeholder="Type or paste your text here...")

# Add a classify button
if st.button('Classify Emotion', type='primary'):
    if text_input:
        with st.spinner('Analyzing text...'):
            # Preprocess the text
            processed_text = preprocess_text(text_input)
            
            # Transform input text
            text_vectorized = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(text_vectorized)
            probabilities = model.predict_proba(text_vectorized)[0]
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('Prediction Results')
                # Map numeric labels to emotion names (updated mapping)
                emotion_map = {
                    0: 'neutral',
                    1: 'sadness',
                    2: 'joy',
                    3: 'fear',
                    4: 'surprise',
                    5: 'anger'
                }
                
                # Display the predicted emotion with an emoji
                emotion_emojis = {
                    'joy': '😊',
                    'sadness': '😢',
                    'anger': '😠',
                    'fear': '😨',
                    'surprise': '😲',
                    'neutral': '😐'
                }
                
                # Get the predicted emotion and confidence
                predicted_emotion = emotion_map[prediction[0]]
                confidence = probabilities[np.argmax(probabilities)]
                
                # Validate the prediction
                final_emotion = validate_emotion(text_input, predicted_emotion, confidence)
                
                # Display results
                st.write(f"### Predicted Emotion: {emotion_emojis.get(final_emotion, '')} {final_emotion.title()}")
                st.write(f"Confidence: {confidence*100:.2f}%")
                
                # Show preprocessing info
                with st.expander("Show Text Analysis"):
                    st.write("Original text:", text_input)
                    st.write("Processed text:", processed_text)
                    st.write("Raw prediction:", prediction[0])
                    st.write("Raw probabilities:", probabilities)
            
            with col2:
                st.subheader('Probability Distribution')
                # Create a DataFrame for the probabilities
                prob_df = pd.DataFrame({
                    'Emotion': [emotion_map[i] for i in range(len(probabilities))],
                    'Probability': probabilities
                })
                # Sort by probability
                prob_df = prob_df.sort_values('Probability', ascending=False)
                
                # Create a bar chart using matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(prob_df['Emotion'], prob_df['Probability'])
                plt.xticks(rotation=45)
                plt.title('Emotion Probabilities')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Add some additional information
            st.subheader('About the Model')
            st.write("""
            This model was trained on the GoEmotions dataset and can classify text into six emotions:
            - Joy 😊
            - Sadness 😢
            - Anger 😠
            - Fear 😨
            - Surprise 😲
            - Neutral 😐
            """)
    else:
        st.warning('Please enter some text to classify.')

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 