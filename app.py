from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import os
import uuid
from feature_extractor import FeatureExtractor
from analyzer import AnalysisResults
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize feature extractor
extractor = FeatureExtractor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        review_text = data.get('review_text', '').strip()
        
        if not review_text:
            return jsonify({'error': 'Review text is required'}), 400
            
        # Analyze the single review
        features = extractor.extract_features(review_text)
        sentiments = extractor.analyze_feature_sentiments(review_text, features)
        
        result = {
            'review': review_text,
            'features': features,
            'sentiments': sentiments
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_csv', methods=['POST'])
def process_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
            
        # Save the uploaded file
        filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the CSV
        df = pd.read_csv(filepath)
        if 'review' not in df.columns:
            return jsonify({'error': "CSV must contain 'review' column"}), 400
            
        # Prepare data for processing
        data = []
        for _, row in df.iterrows():
            data.append({
                'id': str(row.get('id', len(data))),
                'review': str(row['review']).strip(),
                'rating': float(row['rating']) if 'rating' in df.columns and pd.notna(row['rating']) else None
            })
        
        # Process reviews in batches
        results = extractor.batch_process(data)
        
        # Generate analysis report
        analyzer = AnalysisResults(results)
        report = analyzer.generate_report()
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'results': results,
            'analysis': report,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'records_processed': len(results)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)