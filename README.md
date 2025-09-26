# Modern Optical Character Recognition (OCR) System

A comprehensive, multi-engine OCR system built with Python that provides high-accuracy text extraction from images using state-of-the-art techniques and modern web interface.

## Features

### Multiple OCR Engines
- **Tesseract OCR**: Traditional, reliable OCR engine
- **EasyOCR**: Modern deep learning-based OCR with multi-language support
- **PaddleOCR**: Fast and accurate OCR optimized for various languages

### Advanced Image Processing
- **Noise Reduction**: Advanced denoising algorithms
- **Deskewing**: Automatic rotation correction
- **Contrast Enhancement**: CLAHE-based contrast improvement
- **Multiple Preprocessing Methods**: Basic, comprehensive, and aggressive preprocessing

### Modern Web Interface
- **Streamlit UI**: Beautiful, responsive web interface
- **Real-time Processing**: Live OCR results with confidence scoring
- **Batch Processing**: Handle multiple images simultaneously
- **Interactive Visualizations**: Charts and statistics for performance analysis

### Database Integration
- **SQLite Database**: Store all OCR results and metadata
- **Performance Tracking**: Monitor processing times and accuracy
- **Export Capabilities**: Download results as CSV/JSON
- **Search Functionality**: Find specific results in the database

### Additional Features
- **Confidence Scoring**: Automatic best result selection
- **Multi-language Support**: English, Spanish, French, German, Italian
- **Error Handling**: Robust error management and logging
- **Performance Metrics**: Detailed processing statistics

## Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR binary (install from [GitHub](https://github.com/tesseract-ocr/tesseract))

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install Tesseract OCR
- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from [GitHub releases](https://github.com/tesseract-ocr/tesseract/releases)

## Quick Start

### Command Line Usage
```bash
python modern_ocr.py
```

### Web Interface
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Usage Examples

### Basic OCR Processing
```python
from modern_ocr import MultiEngineOCR
import cv2

# Initialize OCR system
ocr = MultiEngineOCR()

# Load image
image = cv2.imread('sample_image.png')

# Extract text with all engines
results = ocr.extract_text_all_engines(image)

# Get best result
best_result = results['best']
print(f"Best text: {best_result['text']}")
print(f"Confidence: {best_result['confidence']:.2f}%")
```

### Advanced Preprocessing
```python
from modern_ocr import AdvancedImagePreprocessor

preprocessor = AdvancedImagePreprocessor()

# Apply comprehensive preprocessing
processed_image = preprocessor.preprocess_image(image, method='comprehensive')

# Use processed image for OCR
results = ocr.extract_text_tesseract(processed_image, preprocess=False)
```

### Database Operations
```python
from modern_ocr import OCRDatabase

# Initialize database
db = OCRDatabase()

# Save results
result_id = db.save_result(image_path, results, metadata)

# Get statistics
stats = db.get_statistics()
print(f"Total processed: {stats['total_results']}")
```

## Project Structure

```
0119_Optical_character_recognition/
├── 0119.py                    # Original OCR implementation
├── modern_ocr.py              # Modern OCR system with multiple engines
├── app.py                     # Streamlit web interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── sample_images/             # Sample images for testing
├── ocr_results.db            # SQLite database (created automatically)
└── .gitignore                # Git ignore file
```

## 🔧 Configuration

### OCR Engine Settings
- **Tesseract**: Configurable PSM and OEM modes
- **EasyOCR**: Multi-language support with confidence filtering
- **PaddleOCR**: Angle classification and language detection

### Preprocessing Options
- **Basic**: Grayscale conversion + thresholding
- **Comprehensive**: Noise reduction + enhancement + deskewing
- **Aggressive**: Multiple denoising + morphological operations

### Database Schema
- **ocr_results**: Main results table with all engine outputs
- **batch_results**: Batch processing tracking

## Performance Metrics

The system tracks various performance metrics:
- Processing time per engine
- Confidence scores
- Text extraction accuracy
- Engine usage statistics
- Batch processing efficiency

## Web Interface Features

### Home Page
- System overview and quick start
- Upload and process single images
- System status and statistics

### Single Image OCR
- Detailed processing options
- Real-time results comparison
- Performance metrics
- Save results to database

### Batch Processing
- Multiple image upload
- Progress tracking
- Results export
- Error handling

### Statistics Dashboard
- Performance analytics
- Engine usage distribution
- Confidence trends
- Recent results table

### Database Management
- Data export capabilities
- Search functionality
- Record management
- Raw data viewing

## 🛠️ Development

### Adding New OCR Engines
1. Create a new method in `MultiEngineOCR` class
2. Implement the engine-specific text extraction
3. Add confidence scoring
4. Update the `extract_text_all_engines` method
5. Add to web interface options

### Custom Preprocessing
1. Add new methods to `AdvancedImagePreprocessor`
2. Implement preprocessing logic
3. Add to preprocessing options
4. Update web interface

### Database Extensions
1. Modify `OCRDatabase` class
2. Add new tables or columns
3. Update save/retrieve methods
4. Add migration scripts if needed

## Testing

### Sample Images
The system includes sample images for testing:
- Clean text images
- Handwritten style text
- Number sequences
- Various fonts and sizes

### Test Scripts
```bash
# Run basic tests
python modern_ocr.py

# Test web interface
streamlit run app.py
```

## Performance Optimization

### Tips for Better Results
1. **Image Quality**: Use high-resolution, clear images
2. **Preprocessing**: Choose appropriate preprocessing method
3. **Engine Selection**: Use multiple engines for comparison
4. **Confidence Filtering**: Filter low-confidence results
5. **Batch Processing**: Process multiple images efficiently

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB+ for large batch processing
- **Storage**: SSD recommended for database operations
- **GPU**: Optional for EasyOCR acceleration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **Tesseract OCR**: Google's open-source OCR engine
- **EasyOCR**: JaidedAI's deep learning OCR library
- **PaddleOCR**: Baidu's OCR toolkit
- **OpenCV**: Computer vision library
- **Streamlit**: Web app framework

## Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed description
4. Include sample images and error logs

## Version History

- **v1.0**: Original Tesseract implementation
- **v2.0**: Multi-engine OCR system
- **v3.0**: Web interface and database integration
- **v4.0**: Advanced preprocessing and batch processing


# Optical-Character-Recognition-System
