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

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

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
        file.save(filepath)
        
        try:
            # Initialize analyzer
            analyzer = FacialSymmetryAnalyzer()
            
            # Process image
            use_perceptual_weights = request.form.get('use_perceptual_weights', 'false').lower() == 'true'
            result = analyzer.analyze_image(filepath, use_perceptual_weights=use_perceptual_weights)
            
            # Save visualization images
            result_id = save_analysis_result(result, filename)
            
            # Save images to static folder
            os.makedirs('static/results', exist_ok=True)
            cv2.imwrite(os.path.join('static/results', f"{result_id}_visualization.png"), result['visualization'])
            cv2.imwrite(os.path.join('static/results', f"{result_id}_left_mirror.png"), result['left_mirror'])
            cv2.imwrite(os.path.join('static/results', f"{result_id}_right_mirror.png"), result['right_mirror'])
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'result_id': result_id,
                'redirect_url': url_for('view_result', result_id=result_id)
            })
            
        except Exception as e:
            # Clean up uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<result_id>')
def download_results(result_id):
    result = get_analysis_result(result_id)
    if not result:
        return render_template('error.html',
                             error_title="Result Not Found",
                             error_message="The requested analysis result could not be found.")
    
    # Create a ZIP file containing all results
    import zipfile
    import io
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        # Add JSON data
        zf.writestr('analysis_data.json', json.dumps(result, indent=2))
        
        # Add visualization images
        zf.write(os.path.join('static/results', f"{result_id}_visualization.png"), 'visualization.png')
        zf.write(os.path.join('static/results', f"{result_id}_left_mirror.png"), 'left_mirror.png')
        zf.write(os.path.join('static/results', f"{result_id}_right_mirror.png"), 'right_mirror.png')
    
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"facial_symmetry_analysis_{result_id}.zip"
    )

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