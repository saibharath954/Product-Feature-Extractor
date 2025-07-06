import argparse
import json
import pandas as pd
from datetime import datetime
from feature_extractor import FeatureExtractor
from analyzer import AnalysisResults
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate input data"""
    try:
        df = pd.read_csv(file_path)
        
        # Basic validation
        if 'review' not in df.columns:
            raise ValueError("Input file must contain 'review' column")
            
        # Convert to list of dicts with standardized keys
        data = []
        for _, row in df.iterrows():
            record = {
                'id': str(row.get('id', len(data))),
                'review': str(row['review']).strip(),
                'rating': row.get('rating', None)
            }
            if record['review']:  # Only include non-empty reviews
                data.append(record)
                
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_results(results: Dict[str, Any], output_path: str):
    """Save results with pretty formatting"""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def print_summary(report: Dict[str, Any]):
    """Print a user-friendly summary"""
    logger.info("\n=== ANALYSIS SUMMARY ===\n")
    
    # Basic stats
    logger.info(f"Total Reviews Processed: {report['summary']['total_reviews']}")
    logger.info(f"Reviews with Features Identified: {report['summary']['reviews_with_features']}")
    logger.info(f"Total Features Extracted: {report['summary']['total_features_identified']}\n")
    
    # Top features
    if report['top_features']:
        logger.info("=== MOST DISCUSSED FEATURES ===")
        for i, feat in enumerate(report['top_features'][:5], 1):
            logger.info(f"{i}. {feat['feature']} (mentioned {feat['count']} times)")
    
    # Strengths
    if report['strengths']:
        logger.info("\n=== PRODUCT STRENGTHS ===")
        for i, strength in enumerate(report['strengths'][:3], 1):
            logger.info(f"{i}. {strength['feature']}")
            logger.info(f"   Positive sentiment: {strength['positive_pct']:.0f}% of mentions")
    
    # Improvement opportunities
    if report['improvement_opportunities']:
        logger.info("\n=== IMPROVEMENT OPPORTUNITIES ===")
        for i, opp in enumerate(report['improvement_opportunities'][:3], 1):
            logger.info(f"{i}. {opp['feature']}")
            logger.info(f"   Negative sentiment: {opp['negative_pct']:.0f}% of mentions")
    
    # Rating correlation
    if 'rating_analysis' in report and report['rating_analysis']:
        logger.info("\n=== HIGHEST RATED FEATURES ===")
        for i, feat in enumerate(report['rating_analysis'][:3], 1):
            logger.info(f"{i}. {feat['feature']} (avg rating: {feat['avg_rating']:.1f})")

def main():
    parser = argparse.ArgumentParser(description="Product Feature Analyzer")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output", default="results.json", help="Output JSON file path")
    args = parser.parse_args()

    try:
        logger.info("\n=== PRODUCT FEATURE ANALYZER ===")
        logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Step 1: Load data
        logger.info("[1/3] Loading and preprocessing data...")
        data = load_data(args.input_file)
        logger.info(f"Loaded {len(data)} valid reviews\n")
        
        # Step 2: Process reviews
        logger.info("[2/3] Extracting features and analyzing sentiments...")
        extractor = FeatureExtractor()
        results = extractor.batch_process(data)
        logger.info(f"Processing complete. Analyzed {len(results)} reviews.\n")
        
        # Step 3: Analyze results
        logger.info("[3/3] Generating insights...")
        analyzer = AnalysisResults(results)
        report = analyzer.generate_report()
        
        # Print and save results
        print_summary(report)
        save_results({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_file': args.input_file,
                'records_processed': len(data)
            },
            'results': results,
            'analysis': report
        }, args.output)
        
        logger.info("\nAnalysis completed successfully!")
        
    except Exception as e:
        logger.error(f"\nProcessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()