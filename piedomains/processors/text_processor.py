#!/usr/bin/env python3
"""
Text processing utilities for domain content analysis.
Handles HTML content extraction, text cleaning, and preprocessing.
"""

import re
import string
from typing import Optional
from bs4 import BeautifulSoup
from bs4.element import Comment

from ..constants import most_common_words
from ..logging import get_logger

logger = get_logger()

# Global variables for NLTK data - will be initialized when needed
words = None
stop_words = None


def _initialize_nltk():
    """Initialize NLTK data with proper error handling."""
    global words, stop_words
    
    if words is not None and stop_words is not None:
        return  # Already initialized
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk.download("stopwords", quiet=True)
        nltk.download("words", quiet=True) 
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        
        # Import and initialize corpora
        from nltk.corpus import stopwords
        words = set(nltk.corpus.words.words())
        stop_words = set(stopwords.words("english"))
        
    except Exception as e:
        logger.warning(f"NLTK initialization failed: {e}")
        # Fallback to basic word sets if NLTK fails
        words = set()
        stop_words = set()


class TextProcessor:
    """Handles text extraction and cleaning from HTML content."""
    
    @staticmethod
    def extract_text_from_html(html_content: str) -> str:
        """
        Extract clean, visible text from HTML content.
        
        Args:
            html_content (str): Raw HTML content
            
        Returns:
            str: Cleaned visible text content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and filter out comments
            texts = soup.find_all(text=True)
            visible_texts = []
            
            for element in texts:
                if element.parent.name not in ['style', 'script', 'meta', 'title']:
                    if not isinstance(element, Comment):
                        text = element.strip()
                        if text:
                            visible_texts.append(text)
            
            return ' '.join(visible_texts)
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    @staticmethod
    def clean_and_normalize_text(text: str) -> str:
        """
        Clean and normalize text data for model input.
        
        Removes numbers, punctuation, non-English words, stopwords, and common terms.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text with English words only, no stopwords or common terms
        """
        if not isinstance(text, str):
            raise AttributeError("Input must be a string")
        
        # Initialize NLTK data if needed
        _initialize_nltk()
        
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        
        # Split into tokens and remove duplicates
        tokens = list(set(text.split()))
        
        # Remove punctuation from each token
        table = str.maketrans("", "", string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        
        # Convert to lowercase and filter alphabetic only
        tokens = [w.lower() for w in tokens if w.isalpha()]
        
        # Remove non-ASCII characters
        tokens = [w for w in tokens if w.isascii()]
        
        # Remove non-English words (only if words corpus is available)
        if words:
            tokens = [w for w in tokens if w in words]
            
        # Filter out stop words
        if stop_words:
            tokens = [w for w in tokens if w not in stop_words]
            
        # Filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        
        # Remove most common generic words
        tokens = [w for w in tokens if w not in most_common_words]
        
        # Remove duplicates again and sort for consistency
        tokens = sorted(list(set(tokens)))
        
        return " ".join(tokens)
    
    @classmethod
    def process_html_to_text(cls, html_content: str) -> str:
        """
        Complete pipeline: extract text from HTML and clean it.
        
        Args:
            html_content (str): Raw HTML content
            
        Returns:
            str: Clean, processed text ready for model input
        """
        # Extract visible text
        raw_text = cls.extract_text_from_html(html_content)
        
        # Clean and normalize
        clean_text = cls.clean_and_normalize_text(raw_text)
        
        return clean_text