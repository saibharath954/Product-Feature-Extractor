import pandas as pd
import logging
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class AnalysisResults:
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.feature_stats = None
        self.sentiment_stats = None
        self.rating_stats = None
        
    def calculate_statistics(self):
        """Calculate comprehensive statistics from results"""
        try:
            feature_counts = defaultdict(int)
            sentiment_counts = defaultdict(lambda: defaultdict(int))
            rating_sums = defaultdict(list)
            
            for result in self.results:
                for feature, sentiment in result['sentiments'].items():
                    feature_counts[feature] += 1
                    sentiment_counts[feature][sentiment] += 1
                    if result['rating'] is not None:
                        rating_sums[feature].append(result['rating'])
            
            self.feature_stats = pd.DataFrame.from_dict(
                feature_counts, orient='index', columns=['count']
            ).sort_values('count', ascending=False)
            
            sentiment_data = []
            for feature, counts in sentiment_counts.items():
                total = sum(counts.values())
                sentiment_data.append({
                    'feature': feature,
                    'total': total,
                    'positive': counts.get('positive', 0),
                    'negative': counts.get('negative', 0),
                    'neutral': counts.get('neutral', 0),
                    'positive_pct': round(counts.get('positive', 0) / total * 100, 1),
                    'negative_pct': round(counts.get('negative', 0) / total * 100, 1),
                    'neutral_pct': round(counts.get('neutral', 0) / total * 100, 1)
                })
            
            self.sentiment_stats = pd.DataFrame(sentiment_data).sort_values('total', ascending=False)
            
            rating_data = []
            for feature, ratings in rating_sums.items():
                rating_data.append({
                    'feature': feature,
                    'avg_rating': round(np.mean(ratings), 2),
                    'rating_count': len(ratings)
                })
            
            self.rating_stats = pd.DataFrame(rating_data).sort_values('avg_rating', ascending=False)
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            raise
                
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        try:
            self.calculate_statistics()
            
            report = {
                'summary': {
                    'total_reviews': len(self.results),
                    'reviews_with_features': sum(1 for r in self.results if r['features']),
                    'total_features_identified': sum(len(r['features']) for r in self.results)
                },
                'top_features': [],
                'feature_sentiments': [],
                'improvement_opportunities': [],
                'strengths': [],
                'rating_analysis': []
            }
            
            if self.feature_stats is not None:
                report['top_features'] = self.feature_stats.head(5).reset_index().rename(
                    columns={'index': 'feature'}).to_dict('records')
            
            if self.sentiment_stats is not None:
                report['feature_sentiments'] = self.sentiment_stats.head(5).to_dict('records')
                
                negative_features = self.sentiment_stats[
                    (self.sentiment_stats['negative_pct'] >= 50) &
                    (self.sentiment_stats['total'] >= 2)
                ].sort_values(['negative_pct', 'total'], ascending=[False, False])
                
                report['improvement_opportunities'] = negative_features.to_dict('records')
                
                positive_features = self.sentiment_stats[
                    (self.sentiment_stats['positive_pct'] >= 50) &
                    (self.sentiment_stats['total'] >= 2)
                ].sort_values(['positive_pct', 'total'], ascending=[False, False])
                
                report['strengths'] = positive_features.to_dict('records')
            
            if self.rating_stats is not None:
                report['rating_analysis'] = self.rating_stats[
                    self.rating_stats['rating_count'] >= 2
                ].to_dict('records')
            
            return report
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise