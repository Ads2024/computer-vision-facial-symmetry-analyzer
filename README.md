# Facial Symmetry Analyzer

A web application that analyzes facial symmetry using computer vision and machine learning techniques. The application provides detailed visualizations and metrics for facial symmetry analysis.

## Features

- Upload and analyze facial images
- Three visualization types:
  - Facial landmarks and alignment
  - Asymmetry heatmap
  - Mirrored comparison
- Detailed symmetry scores for different facial zones
- Perceptual weighting for more accurate analysis
- Modern, responsive web interface
- Drag-and-drop file upload

## Requirements

- Python 3.8 or higher
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- Flask
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/facial-symmetry-analyzer.git
cd facial-symmetry-analyzer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload a facial image using the web interface

4. View the analysis results, including:
   - Overall symmetry score
   - Zone-specific scores
   - Visualizations of facial landmarks
   - Asymmetry heatmap
   - Mirrored comparison

## How It Works

The application uses MediaPipe Face Mesh to detect facial landmarks and calculates symmetry metrics across different facial zones:

- **Face Alignment**: Automatically aligns the face based on eye positions
- **Landmark Detection**: Identifies key facial points using MediaPipe
- **Symmetry Calculation**: Computes symmetry scores for different facial zones
- **Visualization**: Generates three types of visualizations for analysis
- **Perceptual Weighting**: Applies importance weights based on human perception research

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
