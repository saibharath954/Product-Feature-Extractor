from dotenv import load_dotenv
import torch

load_dotenv()

class Config:
    # Model configuration
    MODEL_NAME = "google/flan-t5-small"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float32
    
    # Generation parameters (optimized for feature extraction)
    MAX_NEW_TOKENS = 50  # Reduced for more concise outputs
    TEMPERATURE = 0.3    # Lower for more deterministic outputs
    TOP_P = 0.85
    TOP_K = 20
    DO_SAMPLE = True
    
    # Processing parameters
    BATCH_SIZE = 4
    PROCESSING_DELAY = 0.3
    
    # Enhanced prompt templates
    FEATURE_EXTRACTION_PROMPT = """
    Extract specific product features from this review as a comma-separated list.
    Only output the features, nothing else.
    Example: battery, camera, display, build quality
    
    Review: {review_text}
    Features:"""
    
    SENTIMENT_PROMPT = """
    For each feature, classify sentiment as positive, negative, or neutral.
    Use exact format: feature:sentiment (one per line)
    
    Features: {features}
    Review: "{review_text}"
    Analysis:"""
    
    # Response processing
    FEATURE_SEPARATOR = ","
    SENTIMENT_SEPARATOR = ":"
    VALID_SENTIMENTS = ["positive", "negative", "neutral"]
    
    # Feature cleaning
    FEATURE_SYNONYMS = {
        "camera quality": "camera",
        "battery life": "battery",
        "screen": "display",
        "build": "build quality"
    }
    
    # Model cache
    CACHE_DIR = "./model_cache"
    USE_CACHE = True
    
    # Debug
    VERBOSE = False