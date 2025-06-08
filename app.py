import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import plotly.express as px
import plotly.graph_objects as go

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Emotion keywords and emojis
EMOTION_DATA = {
    'Joy': {
        'keywords': ['happy', 'joy', 'delighted', 'pleased', 'glad', 'cheerful', 'excited', 'thrilled', 'wonderful', 'great', 'amazing', 'fantastic', 'love', 'enjoy', 'beautiful', 'perfect', 'best', 'awesome', 'good', 'nice'],
        'emoji': '😊',
        'color': '#4CAF50'
    },
    'Sadness': {
        'keywords': ['sad', 'unhappy', 'depressed', 'miserable', 'gloomy', 'down', 'blue', 'heartbroken', 'disappointed', 'upset', 'hurt', 'crying', 'tears', 'sorry', 'regret', 'pain', 'loss', 'missing', 'alone', 'empty'],
        'emoji': '😢',
        'color': '#2196F3'
    },
    'Anger': {
        'keywords': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'outraged', 'enraged', 'hate', 'disgust', 'rage', 'fury', 'bitter', 'hostile', 'resentful', 'fuming', 'livid', 'infuriated', 'disgusted', 'hateful'],
        'emoji': '😠',
        'color': '#F44336'
    },
    'Fear': {
        'keywords': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous', 'fearful', 'panic', 'horror', 'dread', 'terror', 'threatened', 'intimidated', 'threat', 'danger', 'unsafe', 'vulnerable', 'threatened', 'anxiety'],
        'emoji': '😨',
        'color': '#9C27B0'
    },
    'Surprise': {
        'keywords': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'unexpected', 'wow', 'incredible', 'unbelievable', 'unexpected', 'sudden', 'unforeseen', 'startled', 'astounded', 'dumbfounded', 'flabbergasted', 'jaw-dropping', 'mind-blowing'],
        'emoji': '😲',
        'color': '#FF9800'
    },
    'Neutral': {
        'keywords': ['okay', 'fine', 'alright', 'normal', 'usual', 'regular', 'average', 'standard', 'typical', 'ordinary', 'common', 'regular', 'moderate', 'balanced', 'stable', 'steady', 'consistent', 'neutral', 'indifferent'],
        'emoji': '😐',
        'color': '#9E9E9E'
    }
}

# Set page config
st.set_page_config(
    page_title="Emotion Classification Model",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #ffffff;
        padding: 2rem;
        min-height: 100vh;
    }
    
    /* Title styling */
    h1 {
        color: #4CAF50;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: fadeIn 1s ease-in;
    }
    
    /* Subheader styling */
    h3 {
        color: #4CAF50;
        font-size: 1.5rem !important;
        margin-top: 1rem;
        animation: slideIn 0.5s ease-out;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(0, 0, 0, 0.2) !important;
        border-radius: 10px !important;
        color: #000000 !important;
        font-size: 1.2rem !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.3) !important;
        background-color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(45deg, #4CAF50, #45a049) !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 0.8rem 2rem !important;
        border: none !important;
        border-radius: 25px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 1rem !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        animation: pulse 2s infinite;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 8px rgba(0,0,0,0.2) !important;
        background: linear-gradient(45deg, #45a049, #4CAF50) !important;
        animation: none;
    }
    
    /* Results container styling */
    .results-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideUp 0.5s ease-out;
    }
    
    /* Emotion card styling */
    .emotion-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .emotion-card:hover {
        transform: translateX(5px);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Success message styling */
    .stSuccess {
        background: rgba(76, 175, 80, 0.2) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin-top: 1rem !important;
        animation: slideUp 0.5s ease-out;
    }
    
    /* Warning message styling */
    .stWarning {
        background: rgba(255, 193, 7, 0.2) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        animation: slideUp 0.5s ease-out;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #888;
        font-size: 0.9rem;
        animation: fadeIn 1s ease-in;
    }
    
    /* Link styling */
    a {
        color: #4CAF50 !important;
        text-decoration: none !important;
        transition: all 0.3s ease;
    }
    
    a:hover {
        text-decoration: underline !important;
        color: #45a049 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("😊 Emotion Classification Model")
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #ffffff;'>
            This application uses machine learning to classify text into different emotions.<br>
            Enter your text below to see the emotion analysis results.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/text_classifier_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Join tokens back into text
    return ' '.join(tokens)

# Keyword-based emotion detection
def get_keyword_emotion(text):
    text = text.lower()
    emotion_scores = {emotion: 0 for emotion in EMOTION_DATA.keys()}
    
    for emotion, data in EMOTION_DATA.items():
        for keyword in data['keywords']:
            if keyword in text:
                emotion_scores[emotion] += 1
    
    return emotion_scores

# Create emotion visualization
def create_emotion_chart(results_df):
    fig = go.Figure()
    
    for _, row in results_df.iterrows():
        emotion = row['Emotion']
        prob = row['Probability']
        color = EMOTION_DATA[emotion]['color']
        emoji = EMOTION_DATA[emotion]['emoji']
        
        fig.add_trace(go.Bar(
            name=f"{emoji} {emotion}",
            x=[emotion],
            y=[prob],
            marker_color=color,
            text=[f"{prob:.1%}"],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Emotion Distribution",
        xaxis_title="Emotions",
        yaxis_title="Probability",
        template="plotly_dark",
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

# Load the model
model, vectorizer = load_model()

# Text input with custom styling
st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;'>
""", unsafe_allow_html=True)
user_input = st.text_area("Enter your text here:", height=150)
st.markdown("</div>", unsafe_allow_html=True)

# Process the input
if st.button("Classify Emotion"):
    if user_input:
        # Preprocess the text
        processed_text = preprocess_text(user_input)
        
        # Get keyword-based emotion scores
        keyword_scores = get_keyword_emotion(user_input)
        
        # Transform the text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Get model predictions
        model_predictions = model.predict_proba(text_vectorized)[0]
        
        # Combine model predictions with keyword scores
        combined_scores = []
        for i, emotion in enumerate(EMOTION_DATA.keys()):
            # Weight the model prediction and keyword score
            model_weight = 0.3
            keyword_weight = 0.7
            
            # If there are any keyword matches, give them more weight
            if keyword_scores[emotion] > 0:
                model_weight = 0.2
                keyword_weight = 0.8
            
            combined_score = (model_predictions[i] * model_weight + 
                            keyword_scores[emotion] * keyword_weight)
            combined_scores.append(combined_score)
        
        # Normalize the combined scores
        total = sum(combined_scores)
        if total > 0:
            combined_scores = [score/total for score in combined_scores]
        
        # Create a DataFrame for better visualization
        results_df = pd.DataFrame({
            'Emotion': list(EMOTION_DATA.keys()),
            'Probability': combined_scores
        })
        
        # Sort by probability
        results_df = results_df.sort_values('Probability', ascending=False)
        
        # Display results in a styled container
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.subheader("Emotion Analysis Results")
        
        # Create columns for the results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Emotions")
            for _, row in results_df.head(3).iterrows():
                emotion = row['Emotion']
                emoji = EMOTION_DATA[emotion]['emoji']
                color = EMOTION_DATA[emotion]['color']
                st.markdown(f"""
                    <div class="emotion-card" style="border-left: 4px solid {color};">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.5rem;">{emoji}</span>
                            <div>
                                <strong style="color: {color};">{emotion}</strong><br>
                                <span>{row['Probability']:.2%}</span>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Emotion Distribution")
            fig = create_emotion_chart(results_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display the most likely emotion
        top_emotion = results_df.iloc[0]
        emoji = EMOTION_DATA[top_emotion['Emotion']]['emoji']
        color = EMOTION_DATA[top_emotion['Emotion']]['color']
        
        # Create columns for the result display
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"<div style='text-align: center; font-size: 3rem;'>{emoji}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='color: {color};'><strong>Most likely emotion: {top_emotion['Emotion']}</strong></div>", unsafe_allow_html=True)
            st.markdown(f"<div>Confidence: {top_emotion['Probability']:.2%}</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.warning("Please enter some text to analyze.")

# Add footer
st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit and scikit-learn</p>
        <p>Check out the <a href='https://github.com/antima121-bit/Emotion-Classification-YOLOv8Model' target='_blank'>GitHub repository</a></p>
    </div>
""", unsafe_allow_html=True) 
