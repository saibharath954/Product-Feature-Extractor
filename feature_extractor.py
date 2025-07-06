from transformers import pipeline, AutoTokenizer
import torch
import logging
import re
from typing import List, Dict, Any
import spacy
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy for better noun phrase extraction
nlp = spacy.load("en_core_web_sm")

class FeatureExtractor:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=self.device,
            torch_dtype=torch.float32
        )
        
        # Common product features to look for
        self.common_features = {
            'camera', 'battery', 'display', 'screen', 'performance', 
            'software', 'build', 'design', 'price', 'audio', 'speaker',
            'charging', 'fingerprint', 'face unlock', 'processor', 'memory',
            'storage', 'gps', 'bluetooth', 'wifi', 'network', 'call quality'
        }
        
        # Feature synonyms mapping
        self.feature_synonyms = {
            'camera quality': 'camera',
            'photo quality': 'camera',
            'battery life': 'battery',
            'screen': 'display',
            'touchscreen': 'display',
            'speed': 'performance',
            'lag': 'performance',
            'ui': 'software',
            'operating system': 'software',
            'construction': 'build',
            'looks': 'design',
            'cost': 'price',
            'sound': 'audio',
            'speakers': 'audio',
            'fast charging': 'charging',
            'fingerprint scanner': 'fingerprint',
            'face recognition': 'face unlock',
            'chip': 'processor',
            'ram': 'memory',
            'storage space': 'storage',
            'navigation': 'gps',
            'wireless': 'wifi',
            'reception': 'network',
            'call clarity': 'call quality'
        }
        
        # Sentiment indicators
        self.positive_words = {
            'excellent', 'great', 'good', 'awesome', 'fantastic', 'amazing',
            'love', 'perfect', 'smooth', 'fast', 'best', 'improved', 'happy'
        }
        
        self.negative_words = {
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst',
            'hate', 'disappointed', 'slow', 'laggy', 'broken', 'issue',
            'problem', 'defective', 'scratch', 'drain', 'overheat'
        }

    def extract_features(self, review_text: str) -> List[str]:
        """Extract product features using combined approach"""
        if not review_text:
            return []
            
        # First try spaCy for noun phrases
        doc = nlp(review_text)
        noun_phrases = set()
        
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            if len(phrase.split()) <= 3:  # Limit to 3-word phrases
                noun_phrases.add(phrase)
        
        # Then use the model for feature extraction
        prompt = f"""
        Extract ONLY specific product features mentioned in this review.
        Respond with a comma-separated list of features only.
        Example: battery life, camera quality, display
        
        Review: {review_text}
        Features:"""
        
        try:
            response = self.model(
                prompt,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                num_return_sequences=1
            )
            
            model_features = set()
            if response and response[0]['generated_text']:
                for feature in response[0]['generated_text'].split(','):
                    feature = feature.strip().lower()
                    if feature:
                        model_features.add(feature)
            
            # Combine both approaches
            all_features = noun_phrases.union(model_features)
            
            # Normalize and filter features
            final_features = []
            for feature in all_features:
                # Normalize using synonyms
                normalized = self._normalize_feature(feature)
                if normalized and normalized not in final_features:
                    final_features.append(normalized)
            
            return final_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return []

    def _normalize_feature(self, feature: str) -> str:
        """Normalize feature names using synonyms and common features"""
        feature = feature.lower().strip()
        
        # Check for direct matches in common features
        for common_feat in self.common_features:
            if common_feat in feature:
                return common_feat
                
        # Check synonyms
        for original, replacement in self.feature_synonyms.items():
            if original in feature:
                return replacement
                
        # If not found in either, return the original if it's a reasonable length
        if 2 <= len(feature.split()) <= 3 and any(char.isalpha() for char in feature):
            return feature
        return ""
    
    def analyze_feature_sentiments(self, review_text: str, features: List[str]) -> Dict[str, str]:
        """Analyze sentiment for each feature using combined approach"""
        if not features or not review_text:
            return {}

        # First try the model for sentiment analysis
        prompt = f"""
        Analyze sentiment for each product feature in this review.
        For each feature, respond with: feature:sentiment
        Sentiment must be one of: positive, negative, or neutral.
        
        Features: {', '.join(features)}
        Review: "{review_text}"
        
        Sentiment Analysis:
        """
        
        model_sentiments = {}
        try:
            response = self.model(
                prompt,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
                num_return_sequences=1
            )
            
            if response and response[0]['generated_text']:
                for line in response[0]['generated_text'].split('\n'):
                    if ':' in line:
                        feature, sentiment = line.split(':', 1)
                        feature = feature.strip().lower()
                        sentiment = sentiment.strip().lower()
                        if feature in features and sentiment in ['positive', 'negative', 'neutral']:
                            model_sentiments[feature] = sentiment
        except Exception as e:
            logger.error(f"Error in model sentiment analysis: {str(e)}")
        
        # Then use rule-based approach as fallback
        rule_sentiments = self._rule_based_sentiment(review_text, features)
        
        # Combine results with model taking precedence
        final_sentiments = {}
        for feature in features:
            final_sentiments[feature] = model_sentiments.get(feature, rule_sentiments.get(feature, 'neutral'))
            
        return final_sentiments
    
    def _rule_based_sentiment(self, review_text: str, features: List[str]) -> Dict[str, str]:
        """Rule-based sentiment analysis as fallback"""
        sentiments = {}
        text_lower = review_text.lower()
        
        for feature in features:
            # Find the context around the feature
            feature_lower = feature.lower()
            pos = text_lower.find(feature_lower)
            if pos == -1:
                sentiments[feature] = 'neutral'
                continue
                
            # Get the surrounding words
            start = max(0, pos - 30)
            end = min(len(text_lower), pos + len(feature) + 30)
            context = text_lower[start:end]
            
            # Check for positive/negative indicators
            positive = sum(1 for word in self.positive_words if word in context)
            negative = sum(1 for word in self.negative_words if word in context)
            
            if positive > negative:
                sentiments[feature] = 'positive'
            elif negative > positive:
                sentiments[feature] = 'negative'
            else:
                sentiments[feature] = 'neutral'
                
        return sentiments
    
    def batch_process(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process reviews in batches"""
        results = []
        
        for review in reviews:
            try:
                text = review.get('review', '')
                if not text:
                    continue
                    
                features = self.extract_features(text)
                if not features:
                    continue
                    
                sentiments = self.analyze_feature_sentiments(text, features)
                
                results.append({
                    'review_id': review.get('id', str(len(results))),
                    'review_text': text,
                    'rating': review.get('rating', None),
                    'features': features,
                    'sentiments': sentiments
                })
                
            except Exception as e:
                logger.error(f"Error processing review {review.get('id')}: {str(e)}")
                continue
                
        return results