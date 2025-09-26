"""
Modern Optical Character Recognition (OCR) System
================================================

This project implements a comprehensive OCR system using multiple engines:
- Tesseract OCR (traditional, reliable)
- EasyOCR (modern, deep learning-based)
- PaddleOCR (fast, accurate for various languages)

Features:
- Multiple OCR engines for comparison
- Advanced image preprocessing
- Confidence scoring
- Batch processing
- Web UI interface
- Database storage for results
"""

import cv2
import numpy as np
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedImagePreprocessor:
    """Advanced image preprocessing for better OCR results"""
    
    def __init__(self):
        self.kernel = np.ones((1, 1), np.uint8)
    
    def preprocess_image(self, image: np.ndarray, method: str = 'comprehensive') -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image as numpy array
            method: Preprocessing method ('basic', 'comprehensive', 'aggressive')
        
        Returns:
            Preprocessed image
        """
        if method == 'basic':
            return self._basic_preprocessing(image)
        elif method == 'comprehensive':
            return self._comprehensive_preprocessing(image)
        elif method == 'aggressive':
            return self._aggressive_preprocessing(image)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    def _basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Basic preprocessing: grayscale + threshold"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _comprehensive_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Comprehensive preprocessing with multiple techniques"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Deskewing
        deskewed = self._deskew_image(enhanced)
        
        # Final threshold
        _, thresh = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _aggressive_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Aggressive preprocessing for difficult images"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Multiple denoising steps
        denoised1 = cv2.fastNlMeansDenoising(gray)
        denoised2 = cv2.medianBlur(denoised1, 3)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(denoised2, cv2.MORPH_CLOSE, kernel)
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return adaptive
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew image by detecting and correcting rotation"""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        if abs(angle) > 0.5:  # Only rotate if angle is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image


class MultiEngineOCR:
    """OCR system using multiple engines for comparison and better accuracy"""
    
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6'
        self.easyocr_reader = None
        self.paddleocr_reader = None
        self.preprocessor = AdvancedImagePreprocessor()
        
        # Initialize engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize OCR engines"""
        try:
            # EasyOCR supports multiple languages
            self.easyocr_reader = easyocr.Reader(['en', 'es', 'fr', 'de', 'it'])
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
        
        try:
            # PaddleOCR
            self.paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
    
    def extract_text_tesseract(self, image: np.ndarray, preprocess: bool = True) -> Dict:
        """Extract text using Tesseract OCR"""
        try:
            if preprocess:
                processed_image = self.preprocessor.preprocess_image(image, 'comprehensive')
            else:
                processed_image = image
            
            # Get text and confidence
            text = pytesseract.image_to_string(processed_image, config=self.tesseract_config)
            
            # Get detailed data
            data = pytesseract.image_to_data(processed_image, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'engine': 'Tesseract',
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'Tesseract', 'error': str(e)}
    
    def extract_text_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text using EasyOCR"""
        if self.easyocr_reader is None:
            return {'text': '', 'confidence': 0, 'engine': 'EasyOCR', 'error': 'EasyOCR not initialized'}
        
        try:
            results = self.easyocr_reader.readtext(image)
            
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    text_parts.append(text)
                    confidences.append(confidence)
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text,
                'confidence': avg_confidence * 100,  # Convert to percentage
                'engine': 'EasyOCR',
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'detections': len(results)
            }
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'EasyOCR', 'error': str(e)}
    
    def extract_text_paddleocr(self, image: np.ndarray) -> Dict:
        """Extract text using PaddleOCR"""
        if self.paddleocr_reader is None:
            return {'text': '', 'confidence': 0, 'engine': 'PaddleOCR', 'error': 'PaddleOCR not initialized'}
        
        try:
            results = self.paddleocr_reader.ocr(image, cls=True)
            
            if not results or not results[0]:
                return {'text': '', 'confidence': 0, 'engine': 'PaddleOCR', 'word_count': 0, 'char_count': 0}
            
            text_parts = []
            confidences = []
            
            for line in results[0]:
                if line[1][1] > 0.5:  # Filter low confidence results
                    text_parts.append(line[1][0])
                    confidences.append(line[1][1])
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text,
                'confidence': avg_confidence * 100,  # Convert to percentage
                'engine': 'PaddleOCR',
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'detections': len(results[0])
            }
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'PaddleOCR', 'error': str(e)}
    
    def extract_text_all_engines(self, image: np.ndarray, preprocess: bool = True) -> Dict:
        """Extract text using all available OCR engines"""
        results = {}
        
        # Tesseract
        results['tesseract'] = self.extract_text_tesseract(image, preprocess)
        
        # EasyOCR
        results['easyocr'] = self.extract_text_easyocr(image)
        
        # PaddleOCR
        results['paddleocr'] = self.extract_text_paddleocr(image)
        
        # Find best result
        best_result = self._find_best_result(results)
        results['best'] = best_result
        
        return results
    
    def _find_best_result(self, results: Dict) -> Dict:
        """Find the best OCR result based on confidence and text length"""
        valid_results = []
        
        for engine, result in results.items():
            if engine != 'best' and 'error' not in result and result['text'].strip():
                valid_results.append(result)
        
        if not valid_results:
            return {'text': '', 'confidence': 0, 'engine': 'None', 'word_count': 0, 'char_count': 0}
        
        # Sort by confidence and text length
        best = max(valid_results, key=lambda x: (x['confidence'], x['char_count']))
        return best


class OCRDatabase:
    """Database for storing OCR results and metadata"""
    
    def __init__(self, db_path: str = "ocr_results.db"):
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # OCR Results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                image_hash TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                preprocessing_method TEXT,
                tesseract_text TEXT,
                tesseract_confidence REAL,
                easyocr_text TEXT,
                easyocr_confidence REAL,
                paddleocr_text TEXT,
                paddleocr_confidence REAL,
                best_text TEXT,
                best_engine TEXT,
                best_confidence REAL,
                processing_time REAL,
                image_width INTEGER,
                image_height INTEGER
            )
        ''')
        
        # Batch Processing table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                success BOOLEAN,
                error_message TEXT,
                processing_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_result(self, image_path: str, results: Dict, metadata: Dict) -> int:
        """Save OCR result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ocr_results (
                image_path, image_hash, preprocessing_method,
                tesseract_text, tesseract_confidence,
                easyocr_text, easyocr_confidence,
                paddleocr_text, paddleocr_confidence,
                best_text, best_engine, best_confidence,
                processing_time, image_width, image_height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_path,
            metadata.get('image_hash', ''),
            metadata.get('preprocessing_method', ''),
            results.get('tesseract', {}).get('text', ''),
            results.get('tesseract', {}).get('confidence', 0),
            results.get('easyocr', {}).get('text', ''),
            results.get('easyocr', {}).get('confidence', 0),
            results.get('paddleocr', {}).get('text', ''),
            results.get('paddleocr', {}).get('confidence', 0),
            results.get('best', {}).get('text', ''),
            results.get('best', {}).get('engine', ''),
            results.get('best', {}).get('confidence', 0),
            metadata.get('processing_time', 0),
            metadata.get('image_width', 0),
            metadata.get('image_height', 0)
        ))
        
        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return result_id
    
    def get_results(self, limit: int = 100) -> pd.DataFrame:
        """Get OCR results from database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f'''
            SELECT * FROM ocr_results 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        ''', conn)
        conn.close()
        return df
    
    def get_statistics(self) -> Dict:
        """Get OCR statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total results
        cursor.execute('SELECT COUNT(*) FROM ocr_results')
        total_results = cursor.fetchone()[0]
        
        # Average confidences
        cursor.execute('SELECT AVG(tesseract_confidence), AVG(easyocr_confidence), AVG(paddleocr_confidence) FROM ocr_results')
        avg_confidences = cursor.fetchone()
        
        # Best engine distribution
        cursor.execute('SELECT best_engine, COUNT(*) FROM ocr_results GROUP BY best_engine')
        engine_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_results': total_results,
            'average_confidences': {
                'tesseract': avg_confidences[0] or 0,
                'easyocr': avg_confidences[1] or 0,
                'paddleocr': avg_confidences[2] or 0
            },
            'engine_distribution': engine_distribution
        }


def create_sample_images():
    """Create sample images for testing OCR"""
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a simple text image
    from PIL import Image, ImageDraw, ImageFont
    
    # Sample 1: Clean text
    img1 = Image.new('RGB', (400, 100), color='white')
    draw1 = ImageDraw.Draw(img1)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw1.text((20, 30), "Hello World! This is a test.", fill='black', font=font)
    img1.save(sample_dir / "clean_text.png")
    
    # Sample 2: Handwritten style
    img2 = Image.new('RGB', (400, 100), color='white')
    draw2 = ImageDraw.Draw(img2)
    draw2.text((20, 30), "Handwritten style text", fill='black', font=font)
    img2.save(sample_dir / "handwritten_style.png")
    
    # Sample 3: Numbers
    img3 = Image.new('RGB', (400, 100), color='white')
    draw3 = ImageDraw.Draw(img3)
    draw3.text((20, 30), "12345 67890", fill='black', font=font)
    img3.save(sample_dir / "numbers.png")
    
    logger.info(f"Sample images created in {sample_dir}")


def main():
    """Main function to demonstrate OCR capabilities"""
    # Create sample images
    create_sample_images()
    
    # Initialize OCR system
    ocr_system = MultiEngineOCR()
    database = OCRDatabase()
    
    # Test with sample images
    sample_images = [
        "sample_images/clean_text.png",
        "sample_images/handwritten_style.png", 
        "sample_images/numbers.png"
    ]
    
    for image_path in sample_images:
        if os.path.exists(image_path):
            print(f"\n{'='*50}")
            print(f"Processing: {image_path}")
            print(f"{'='*50}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue
            
            # Process with all engines
            start_time = datetime.now()
            results = ocr_system.extract_text_all_engines(image)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            for engine, result in results.items():
                if engine != 'best':
                    print(f"\n{engine.upper()} Results:")
                    print(f"Text: {result.get('text', 'N/A')}")
                    print(f"Confidence: {result.get('confidence', 0):.2f}%")
                    if 'error' in result:
                        print(f"Error: {result['error']}")
            
            print(f"\nBEST RESULT:")
            best = results['best']
            print(f"Engine: {best.get('engine', 'N/A')}")
            print(f"Text: {best.get('text', 'N/A')}")
            print(f"Confidence: {best.get('confidence', 0):.2f}%")
            
            # Save to database
            metadata = {
                'image_hash': str(hash(image_path)),
                'preprocessing_method': 'comprehensive',
                'processing_time': processing_time,
                'image_width': image.shape[1],
                'image_height': image.shape[0]
            }
            
            result_id = database.save_result(image_path, results, metadata)
            print(f"Saved to database with ID: {result_id}")
    
    # Display statistics
    stats = database.get_statistics()
    print(f"\n{'='*50}")
    print("DATABASE STATISTICS")
    print(f"{'='*50}")
    print(f"Total results: {stats['total_results']}")
    print(f"Average confidences:")
    for engine, conf in stats['average_confidences'].items():
        print(f"  {engine}: {conf:.2f}%")
    print(f"Engine distribution: {stats['engine_distribution']}")


if __name__ == "__main__":
    main()
