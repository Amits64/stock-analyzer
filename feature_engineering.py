import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

# Download necessary NLTK resources if not already downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize NLTK components for text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Preprocesses input text for sentiment analysis.

    Parameters:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove stopwords and lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def extract_social_media_sentiment_features(symbol, text):
    """
    Extract sentiment features from social media discussions using VADER sentiment analysis.

    Parameters:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC' for Bitcoin).
        text (str): Text to analyze sentiment.

    Returns:
        dict: Dictionary containing sentiment features.
    """
    # Preprocess text
    preprocessed_text = preprocess_text(text)

    # Analyze sentiment using VADER
    sentiment_scores = sentiment_analyzer.polarity_scores(preprocessed_text)

    # Construct sentiment features dictionary
    social_media_sentiment_features = {
        'symbol': symbol,
        'positive_sentiment': sentiment_scores['pos'],
        'negative_sentiment': sentiment_scores['neg'],
        'neutral_sentiment': sentiment_scores['neu'],
        'compound_sentiment': sentiment_scores['compound']
    }

    return social_media_sentiment_features
