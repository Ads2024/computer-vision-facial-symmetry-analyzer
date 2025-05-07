import os
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
from facial_symmetry_analyzer import FacialSymmetryAnalyzer
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive mode
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime
import uuid
import cv2
import numpy as np
import tempfile
import shutil
import glob
import zipfile

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
STATIC_RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['STATIC_RESULTS_FOLDER'] = STATIC_RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(STATIC_RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_numpy_to_json(obj):
    """Convert NumPy arrays and other non-JSON serializable objects to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_json(item) for item in obj]
    return obj

def save_analysis_result(result_data, filename):
    """Save analysis results to a JSON file"""
    result_id = str(uuid.uuid4())
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    
    # Convert NumPy arrays to JSON-serializable format
    serializable_data = convert_numpy_to_json(result_data)
    
    # Add metadata
    serializable_data.update({
        'id': result_id,
        'filename': filename,
        'timestamp': datetime.now().isoformat()
    })
    
    with open(result_path, 'w') as f:
        json.dump(serializable_data, f)
    
    return result_id

def get_analysis_result(result_id):
    """Retrieve analysis results from JSON file"""
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            return json.load(f)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
    # Get all analysis results
    analyses = []
    for filename in os.listdir(app.config['RESULTS_FOLDER']):
        if filename.endswith('.json'):
            with open(os.path.join(app.config['RESULTS_FOLDER'], filename), 'r') as f:
                analyses.append(json.load(f))
    
    # Sort by timestamp, newest first
    analyses.sort(key=lambda x: x['timestamp'], reverse=True)
    return render_template('history.html', analyses=analyses)

@app.route('/view/<result_id>')
def view_result(result_id):
    result = get_analysis_result(result_id)
    if not result:
        return render_template('error.html', 
                             error_title="Result Not Found",
                             error_message="The requested analysis result could not be found.")
    
    # Generate URLs for visualization images
    visualization_url = url_for('static', filename=f'results/{result_id}_visualization.png')
    left_mirror_url = url_for('static', filename=f'results/{result_id}_left_mirror.png')
    right_mirror_url = url_for('static', filename=f'results/{result_id}_right_mirror.png')
    
    return render_template('result.html',
                         result=result,
                         result_id=result_id,
                         visualization_url=visualization_url,
                         left_mirror_url=left_mirror_url,
                         right_mirror_url=right_mirror_url)

@app.route('/compare')
def compare_results():
    result1_id = request.args.get('result1')
    result2_id = request.args.get('result2')
    
    if not result1_id or not result2_id:
        return render_template('error.html',
                             error_title="Invalid Comparison",
                             error_message="Please select two analyses to compare.")
    
    result1 = get_analysis_result(result1_id)
    result2 = get_analysis_result(result2_id)
    
    if not result1 or not result2:
        return render_template('error.html',
                             error_title="Result Not Found",
                             error_message="One or both of the selected analyses could not be found.")
    
    return render_template('compare.html',
                         result1=result1,
                         result2=result2)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save the uploaded file
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Initialize analyzer
            analyzer = FacialSymmetryAnalyzer()
            
            # Process image
            use_perceptual_weights = request.form.get('use_perceptual_weights', 'false').lower() == 'true'
            print(f"Processing image with perceptual weights: {use_perceptual_weights}")
            
            try:
                result = analyzer.analyze_image(filepath, use_perceptual_weights=use_perceptual_weights)
                print("Image analysis completed successfully")
            except Exception as e:
                print(f"Error during image analysis: {str(e)}")
                raise
            
            # Save visualization images
            try:
                result_id = save_analysis_result(result, filename)
                print(f"Analysis results saved with ID: {result_id}")
            except Exception as e:
                print(f"Error saving analysis results: {str(e)}")
                raise
            
            # Save images to static folder
            try:
                vis_path = os.path.join(app.config['STATIC_RESULTS_FOLDER'], f"{result_id}_visualization.png")
                left_mirror_path = os.path.join(app.config['STATIC_RESULTS_FOLDER'], f"{result_id}_left_mirror.png")
                right_mirror_path = os.path.join(app.config['STATIC_RESULTS_FOLDER'], f"{result_id}_right_mirror.png")
                
                cv2.imwrite(vis_path, result['visualization'])
                cv2.imwrite(left_mirror_path, result['left_mirror'])
                cv2.imwrite(right_mirror_path, result['right_mirror'])
                print("Visualization images saved successfully")
            except Exception as e:
                print(f"Error saving visualization images: {str(e)}")
                raise
            
            # Clean up uploaded file
            os.remove(filepath)
            print("Temporary file cleaned up")
            
            return jsonify({
                'success': True,
                'result_id': result_id,
                'redirect_url': url_for('view_result', result_id=result_id)
            })
            
        except Exception as e:
            # Clean up uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': f'An error occurred while processing the image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<result_id>')
def download_result(result_id):
    try:
        result = get_analysis_result(result_id)
        if not result:
            return jsonify({'error': 'Result not found'}), 404

        # Create a temporary directory for the files
        temp_dir = tempfile.mkdtemp()
        try:
            # Save the JSON data
            json_path = os.path.join(temp_dir, 'analysis.json')
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)

            # Copy the visualization images
            vis_path = os.path.join(app.config['STATIC_RESULTS_FOLDER'], f"{result_id}_visualization.png")
            if os.path.exists(vis_path):
                shutil.copy2(vis_path, os.path.join(temp_dir, 'visualization.png'))

            # Create the ZIP file
            zip_path = os.path.join(temp_dir, f'analysis_{result_id}.zip')
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(json_path, 'analysis.json')
                if os.path.exists(vis_path):
                    zipf.write(os.path.join(temp_dir, 'visualization.png'), 'visualization.png')

            return send_file(zip_path, as_attachment=True, download_name=f'analysis_{result_id}.zip')
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error downloading result: {str(e)}")
        return jsonify({'error': 'Error downloading result'}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    try:
        # Get all result files
        result_files = glob.glob(os.path.join(app.config['RESULTS_FOLDER'], '*.json'))
        
        # Delete each result file and its associated visualization
        for result_file in result_files:
            result_id = os.path.splitext(os.path.basename(result_file))[0]
            
            # Delete the JSON file
            os.remove(result_file)
            
            # Delete the visualization image if it exists
            vis_path = os.path.join(app.config['STATIC_RESULTS_FOLDER'], f"{result_id}_visualization.png")
            if os.path.exists(vis_path):
                os.remove(vis_path)
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error clearing history: {str(e)}")
        return jsonify({'error': 'Error clearing history'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html',
                         error_title="Page Not Found",
                         error_message="The requested page could not be found."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html',
                         error_title="Internal Server Error",
                         error_message="An unexpected error occurred."), 500

if __name__ == '__main__':
    app.run(debug=True) 