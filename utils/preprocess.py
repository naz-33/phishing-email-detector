"""
Preprocessing functions for both Baseline and Hybrid models
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def get_wordnet_pos(nltk_tag):
    """Map NLTK POS tags to WordNet POS tags."""
    if nltk_tag.startswith('V'):
        return 'v'
    return 'n'


def preprocess_email_baseline(text):
    """
    Preprocess email for Baseline Model (Logistic Regression).
    Steps: HTML removal, URL removal, email removal, numeric removal,
           special char removal, lowercase, tokenize, stopword removal,
           lemmatization.
    """
    text = str(text)
    
    # 1. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 2. Remove HTTP/HTTPS URLs
    text = re.sub(r"https?://\S+", " ", text)
    
    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    
    # 4. Remove numeric sequences
    text = re.sub(r"\d+", " ", text)
    
    # 5. Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # 7. Lowercase
    text = text.lower()
    
    # 8. Tokenize
    tokens = text.split()
    
    # 9. Remove NLTK English stop words
    tokens = [token for token in tokens if token not in STOP_WORDS]
    
    # 10. Lemmatize noun, then verb
    tokens = [
        LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(token, pos="n"), pos="v")
        for token in tokens
    ]
    
    return " ".join(tokens)


def preprocess_email_hybrid(text):
    """
    Preprocess email for Hybrid Model (LR + LightGBM).
    Follows Table 3 Steps 1-10 (full preprocessing for TF-IDF).
    """
    text = str(text)
    
    # Step 1: HTML removal
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Step 2: URL removal
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Step 3: Email removal
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Step 4: Raw corpus preservation (done separately for stylometrics)
    
    # Step 5: Special char removal
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Step 6: Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 7: Lowercasing
    text = text.lower()
    
    # Step 8: Tokenization
    try:
        tokens = nltk.word_tokenize(text)
    except:
        tokens = text.split()
    
    # Step 9: Stop word removal
    tokens = [t for t in tokens if t not in STOP_WORDS]
    
    # Step 10: Lemmatization with POS tagging
    try:
        tagged = nltk.pos_tag(tokens)
        tokens = [
            LEMMATIZER.lemmatize(tok, get_wordnet_pos(tag))
            for tok, tag in tagged
        ]
    except:
        tokens = [LEMMATIZER.lemmatize(tok) for tok in tokens]
    
    return ' '.join(tokens)


def clean_for_stylometrics(text):
    """Minimal cleaning for stylometric features (preserves URLs, case, punctuation)."""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()