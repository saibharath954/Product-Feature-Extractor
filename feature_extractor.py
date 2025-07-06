from transformers import pipeline, AutoTokenizer
from config import Config
import time
from tqdm import tqdm
import torch
import logging
import re
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.config = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.pipeline = self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize a single pipeline for both tasks"""
        try:
            logger.info("Loading FLAN-T5 model...")
            return pipeline(
                "text2text-generation",
                model=self.config.MODEL_NAME,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=self.config.TORCH_DTYPE,
                tokenizer=self.tokenizer,
                model_kwargs={"cache_dir": self.config.CACHE_DIR}
            )
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
        
    def extract_features(self, review_text: str) -> List[str]:
        """Extract and clean product features"""
        if not review_text:
            return []
            
        prompt = """
        Extract ONLY the specific product features mentioned in this review.
        Respond with a comma-separated list of noun phrases only.
        Example: battery life, camera quality, screen resolution
        
        Review: {review_text}
        Features:""".format(review_text=review_text).strip()
        
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=100,
                temperature=0.1,  # More deterministic
                top_p=0.9,
                top_k=20,
                do_sample=False,
                num_return_sequences=1
            )
            
            # Clean and normalize the features
            raw_features = response[0]['generated_text'].strip()
            features = []
            
            for feature in re.split(r'[,;]', raw_features):
                feature = feature.strip().lower()
                
                # Remove anything after 'but', 'though', etc.
                feature = re.split(r'\b(but|though|however)\b', feature)[0].strip()
                
                # Remove any descriptive clauses
                feature = re.sub(r'\b(with|that|which|when|where)\b.*', '', feature).strip()
                
                # Remove special characters except spaces
                feature = re.sub(r'[^a-zA-Z0-9\s]', '', feature).strip()
                
                # Normalize common features
                feature = self._normalize_feature(feature)
                
                if feature and len(feature.split()) <= 4 and feature not in features:  # Limit to 4-word phrases
                    features.append(feature)
                    
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return []

    def _normalize_feature(self, feature: str) -> str:
        """Normalize feature names using synonyms"""
        feature = feature.lower()
        
        # Common substitutions
        substitutions = {
            'battery life': 'battery',
            'cam': 'camera',
            'display': 'screen',
            'build': 'build quality',
            'speaker': 'audio',
            'sound': 'audio',
            'charging speed': 'charging',
            'fingerprint sensor': 'fingerprint'
        }
        
        for original, replacement in substitutions.items():
            if original in feature:
                return replacement
                
        return feature
    
    def analyze_feature_sentiments(self, review_text: str, features: List[str]) -> Dict[str, str]:
        """Analyze sentiment for each feature in the review with error handling"""
        if not features or not review_text:
            return {}

        # Create a more structured prompt for better sentiment analysis
        prompt = """
        Analyze the sentiment for each product feature mentioned in this review.
        For each feature, respond with exactly: feature:sentiment
        Sentiment must be one of: positive, negative, or neutral.
        
        Features to analyze: {features}
        
        Review: "{review_text}"
        
        Sentiment Analysis:
        """.format(
            review_text=review_text,
            features=", ".join(features)
        ).strip()

        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=150,  # Increased for multiple features
                temperature=0.1,  # Lower for more consistent results
                top_p=0.9,
                top_k=20,
                do_sample=False,  # Disable sampling for more deterministic results
                num_return_sequences=1
            )
            
            # Parse the response more carefully
            sentiments = {}
            for line in response[0]['generated_text'].split('\n'):
                line = line.strip()
                if ':' in line:
                    feature_part, sentiment_part = line.split(':', 1)
                    feature = feature_part.strip().lower()
                    sentiment = sentiment_part.strip().lower()
                    
                    # Validate the sentiment
                    if sentiment in self.config.VALID_SENTIMENTS:
                        # Find which original feature this matches best
                        for orig_feature in features:
                            if orig_feature.lower() in feature or feature in orig_feature.lower():
                                sentiments[orig_feature] = sentiment
                                break
            
            # Fallback for any features not detected
            for feature in features:
                if feature not in sentiments:
                    # Try to determine sentiment from review text
                    if any(neg_word in review_text.lower() for neg_word in ['not', 'no', 'lack', 'poor', 'bad']):
                        sentiments[feature] = 'negative'
                    elif any(pos_word in review_text.lower() for pos_word in ['good', 'great', 'excellent', 'love']):
                        sentiments[feature] = 'positive'
                    else:
                        sentiments[feature] = 'neutral'
            
            return sentiments
            
        except Exception as e:
            logger.error(f"Error analyzing sentiments: {str(e)}")
            # Fallback: assign neutral to all features
            return {feature: "neutral" for feature in features}
    
    def _clean_features(self, text: str) -> List[str]:
        """Clean and normalize extracted features"""
        if not text:
            return []
            
        # Basic cleaning
        features = []
        for item in re.split(r'[,;\n]', text):
            item = item.strip().lower()
            if not item:
                continue
                
            # Remove anything in parentheses and special characters
            item = re.sub(r'\([^)]*\)', '', item)
            item = re.sub(r'[^a-zA-Z0-9\s]', '', item).strip()
            
            # Normalize common terms
            for syn, normalized in self.config.FEATURE_SYNONYMS.items():
                if syn in item:
                    item = normalized
                    break
                    
            if item and item not in features:
                features.append(item)
                
        return features
    
    def _parse_sentiments(self, text: str) -> Dict[str, str]:
        """Parse sentiment analysis results with validation"""
        sentiments = {}
        for line in text.split('\n'):
            if self.config.SENTIMENT_SEPARATOR in line:
                parts = line.split(self.config.SENTIMENT_SEPARATOR, 1)
                if len(parts) == 2:
                    feature, sentiment = parts
                    feature = feature.strip().lower()
                    sentiment = sentiment.strip().lower()
                    
                    # Validate sentiment
                    if sentiment not in self.config.VALID_SENTIMENTS:
                        sentiment = "neutral"
                        
                    if feature:
                        sentiments[feature] = sentiment
                        
        return sentiments
    
    def batch_process(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process reviews in batches with progress tracking"""
        if not reviews:
            return []
            
        results = []
        total = len(reviews)
        
        for i in tqdm(range(0, total, self.config.BATCH_SIZE), desc="Processing reviews"):
            batch = reviews[i:i + self.config.BATCH_SIZE]
            
            for review in batch:
                try:
                    text = review.get('review', '')
                    if not text:
                        continue
                        
                    features = self.extract_features(text)
                    if not features:
                        continue
                        
                    sentiments = self.analyze_feature_sentiments(text, features)
                    
                    results.append({
                        'review_id': review.get('id', str(i)),
                        'review_text': text,
                        'rating': review.get('rating', None),
                        'features': features,
                        'sentiments': sentiments
                    })
                except Exception as e:
                    logger.error(f"Error processing review: {str(e)}")
                    continue
                
            time.sleep(self.config.PROCESSING_DELAY)
            
        return results