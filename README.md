# Emotional Classification Model

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.32.0-red" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/scikit--learn-1.4.0-orange" alt="scikit-learn Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

## 🎥 Video Demonstration

Watch a detailed walkthrough of the project:
[![Project Demo](https://img.shields.io/badge/Watch-Demo-red)](https://www.loom.com/share/48f0331b71f7416581624473062d0732?sid=4faf3e6d-7149-434d-90f2-d61cdf3beb8a)

The video covers:
- Project overview and features
- Installation process
- Live demonstration of emotion classification
- Example use cases
- Technical implementation details

## 📝 Overview

This project implements a text emotion classification system using machine learning. It can analyze text input and classify it into six different emotions: Joy, Sadness, Anger, Fear, Surprise, and Neutral. The system uses the GoEmotions dataset for training and provides an interactive web interface built with Streamlit.

## 📊 Dataset

The project uses the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) from Google Research, which is a large-scale dataset of Reddit comments labeled with 27 emotions. For this project, we focus on the following emotions:

- Joy
- Sadness
- Anger
- Fear
- Surprise
- Neutral

Dataset Statistics:
- Total samples: 58,000+
- Average text length: 50-100 words
- Multi-label annotations
- Balanced class distribution

## 🧠 Technical Approach

### 1. Data Preprocessing
- Text cleaning and normalization
- Removal of special characters and URLs
- Conversion to lowercase
- Handling of contractions and slang
- Tokenization and lemmatization

### 2. Feature Engineering
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- N-gram features (unigrams and bigrams)
- Custom feature extraction for emotion-specific keywords

### 3. Model Architecture
- Multi-class classification using scikit-learn
- Support Vector Machine (SVM) classifier
- Class weighting to handle imbalanced data
- Cross-validation for model evaluation

### 4. Validation System
- Keyword-based validation for improved accuracy
- Confidence thresholding
- Ensemble approach combining model predictions with keyword matching

## 📦 Dependencies

The project requires the following Python packages:

```python
# Core Dependencies
streamlit==1.32.0        # Web interface
scikit-learn==1.4.0      # Machine learning
pandas==2.2.0            # Data manipulation
numpy==1.26.3            # Numerical computations
matplotlib==3.8.2        # Data visualization
joblib==1.3.2            # Model persistence

# Optional Dependencies
nltk==3.8.1             # Natural Language Processing
spacy==3.7.2            # Advanced NLP
```

## ✨ Features

- **Multi-Emotion Classification**: Classifies text into 6 distinct emotions:
  - 😊 Joy
  - 😢 Sadness
  - 😠 Anger
  - 😨 Fear
  - 😲 Surprise
  - 😐 Neutral

- **Interactive Web Interface**: User-friendly Streamlit interface for real-time emotion analysis
- **Confidence Scoring**: Provides confidence scores for each emotion prediction
- **Visual Analytics**: Displays probability distribution across all emotions
- **Text Preprocessing**: Implements advanced text cleaning and validation
- **Keyword-Based Validation**: Enhances accuracy through keyword matching

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/antima121-bit/GoEmotions-Text-Emotion-Classification.git
cd GoEmotions-Text-Emotion-Classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter your text in the input field and click "Classify Emotion" to see the results

## 📁 Project Structure

```
GoEmotions-Text-Emotion-Classification/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT License
├── CONTRIBUTING.md       # Contribution guidelines
├── .gitignore           # Git ignore rules
└── models/              # Directory containing trained models
    ├── text_classifier_model.pkl
    └── tfidf_vectorizer.pkl
```

## 🧠 Model Details

- **Dataset**: Trained on the GoEmotions dataset
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Validation**: Keyword-based validation system
- **Output**: Probability distribution across emotion categories

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Antima** - [@antima121-bit](https://github.com/antima121-bit)

## 🙏 Acknowledgments

- GoEmotions dataset for training data
- Streamlit for the web interface framework
- scikit-learn for machine learning capabilities
