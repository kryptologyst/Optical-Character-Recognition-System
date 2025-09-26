#!/usr/bin/env python3
"""
Batch Processing Script for Modern OCR System
============================================

This script provides command-line batch processing capabilities for the OCR system.
It can process multiple images from a directory and save results to the database.

Usage:
    python batch_process.py --input_dir /path/to/images --output_dir /path/to/results
    python batch_process.py --input_dir /path/to/images --engines tesseract easyocr
    python batch_process.py --help
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Optional

# Import our OCR system
from modern_ocr import MultiEngineOCR, OCRDatabase, AdvancedImagePreprocessor
import cv2
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processing class for OCR operations"""
    
    def __init__(self, engines: List[str] = None, preprocessing_method: str = 'comprehensive'):
        """
        Initialize batch processor
        
        Args:
            engines: List of OCR engines to use
            preprocessing_method: Preprocessing method to apply
        """
        self.engines = engines or ['tesseract', 'easyocr', 'paddleocr']
        self.preprocessing_method = preprocessing_method
        self.ocr_system = MultiEngineOCR()
        self.database = OCRDatabase()
        self.preprocessor = AdvancedImagePreprocessor()
        
        # Supported image formats
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'failed': 0,
            'total_time': 0,
            'start_time': None,
            'results': []
        }
    
    def get_image_files(self, input_dir: str) -> List[Path]:
        """Get all image files from input directory"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
        
        image_files = []
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        logger.info(f"Found {len(image_files)} image files in {input_dir}")
        return image_files
    
    def process_single_image(self, image_path: Path) -> Dict:
        """Process a single image with OCR"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Process with selected engines
            results = {}
            processing_times = {}
            
            start_time = time.time()
            
            if 'tesseract' in self.engines:
                t_start = time.time()
                results['tesseract'] = self.ocr_system.extract_text_tesseract(
                    image, preprocess=(self.preprocessing_method != 'basic')
                )
                processing_times['tesseract'] = time.time() - t_start
            
            if 'easyocr' in self.engines:
                t_start = time.time()
                results['easyocr'] = self.ocr_system.extract_text_easyocr(image)
                processing_times['easyocr'] = time.time() - t_start
            
            if 'paddleocr' in self.engines:
                t_start = time.time()
                results['paddleocr'] = self.ocr_system.extract_text_paddleocr(image)
                processing_times['paddleocr'] = time.time() - t_start
            
            total_time = time.time() - start_time
            
            # Find best result
            best_result = self.ocr_system._find_best_result(results)
            results['best'] = best_result
            
            # Prepare result data
            result_data = {
                'image_path': str(image_path),
                'success': True,
                'processing_time': total_time,
                'processing_times': processing_times,
                'results': results,
                'best_text': best_result.get('text', ''),
                'best_engine': best_result.get('engine', ''),
                'best_confidence': best_result.get('confidence', 0),
                'image_width': image.shape[1],
                'image_height': image.shape[0],
                'timestamp': datetime.now()
            }
            
            logger.info(f"Processed {image_path.name}: {best_result.get('engine', 'N/A')} "
                       f"(confidence: {best_result.get('confidence', 0):.2f}%)")
            
            return result_data
            
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {str(e)}")
            return {
                'image_path': str(image_path),
                'success': False,
                'error': str(e),
                'processing_time': 0,
                'timestamp': datetime.now()
            }
    
    def process_batch(self, input_dir: str, save_to_db: bool = True, 
                     export_results: bool = True, output_dir: str = None) -> Dict:
        """
        Process a batch of images
        
        Args:
            input_dir: Directory containing images
            save_to_db: Whether to save results to database
            export_results: Whether to export results to files
            output_dir: Directory to save exported results
        
        Returns:
            Dictionary with processing statistics and results
        """
        self.stats['start_time'] = datetime.now()
        
        # Get image files
        image_files = self.get_image_files(input_dir)
        self.stats['total_images'] = len(image_files)
        
        if not image_files:
            logger.warning("No image files found in input directory")
            return self.stats
        
        logger.info(f"Starting batch processing of {len(image_files)} images")
        logger.info(f"Using engines: {', '.join(self.engines)}")
        logger.info(f"Preprocessing method: {self.preprocessing_method}")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            result = self.process_single_image(image_path)
            self.stats['results'].append(result)
            
            if result['success']:
                self.stats['processed'] += 1
                self.stats['total_time'] += result['processing_time']
                
                # Save to database if requested
                if save_to_db:
                    try:
                        metadata = {
                            'image_hash': str(hash(str(image_path))),
                            'preprocessing_method': self.preprocessing_method,
                            'processing_time': result['processing_time'],
                            'image_width': result['image_width'],
                            'image_height': result['height']
                        }
                        
                        result_id = self.database.save_result(
                            str(image_path), result['results'], metadata
                        )
                        logger.debug(f"Saved to database with ID: {result_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to save to database: {e}")
            else:
                self.stats['failed'] += 1
        
        # Export results if requested
        if export_results:
            self.export_results(output_dir)
        
        # Calculate final statistics
        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        logger.info(f"Batch processing completed:")
        logger.info(f"  Total images: {self.stats['total_images']}")
        logger.info(f"  Processed: {self.stats['processed']}")
        logger.info(f"  Failed: {self.stats['failed']}")
        logger.info(f"  Total time: {self.stats['total_time']:.2f}s")
        logger.info(f"  Duration: {self.stats['duration']:.2f}s")
        
        return self.stats
    
    def export_results(self, output_dir: str = None):
        """Export processing results to files"""
        if not output_dir:
            output_dir = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export CSV
        csv_data = []
        for result in self.stats['results']:
            if result['success']:
                csv_data.append({
                    'image_path': result['image_path'],
                    'best_engine': result['best_engine'],
                    'best_confidence': result['best_confidence'],
                    'best_text': result['best_text'],
                    'processing_time': result['processing_time'],
                    'image_width': result['image_width'],
                    'image_height': result['image_height'],
                    'timestamp': result['timestamp']
                })
            else:
                csv_data.append({
                    'image_path': result['image_path'],
                    'best_engine': 'N/A',
                    'best_confidence': 0,
                    'best_text': '',
                    'processing_time': 0,
                    'image_width': 0,
                    'image_height': 0,
                    'timestamp': result['timestamp'],
                    'error': result.get('error', '')
                })
        
        df = pd.DataFrame(csv_data)
        csv_file = output_path / 'batch_results.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"Results exported to CSV: {csv_file}")
        
        # Export detailed JSON
        import json
        json_file = output_path / 'batch_results.json'
        with open(json_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        logger.info(f"Detailed results exported to JSON: {json_file}")
        
        # Export statistics summary
        summary_file = output_path / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Batch Processing Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Start time: {self.stats['start_time']}\n")
            f.write(f"End time: {self.stats['end_time']}\n")
            f.write(f"Duration: {self.stats['duration']:.2f} seconds\n")
            f.write(f"Total images: {self.stats['total_images']}\n")
            f.write(f"Processed: {self.stats['processed']}\n")
            f.write(f"Failed: {self.stats['failed']}\n")
            f.write(f"Success rate: {(self.stats['processed']/self.stats['total_images']*100):.2f}%\n")
            f.write(f"Total processing time: {self.stats['total_time']:.2f} seconds\n")
            f.write(f"Average processing time: {(self.stats['total_time']/self.stats['processed']):.2f} seconds\n")
        
        logger.info(f"Summary exported to: {summary_file}")


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description='Batch process images with OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process.py --input_dir ./images
  python batch_process.py --input_dir ./images --engines tesseract easyocr
  python batch_process.py --input_dir ./images --output_dir ./results --no-db
  python batch_process.py --input_dir ./images --preprocessing aggressive
        """
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        required=True,
        help='Input directory containing images'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        help='Output directory for results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--engines', '-e',
        nargs='+',
        choices=['tesseract', 'easyocr', 'paddleocr'],
        default=['tesseract', 'easyocr', 'paddleocr'],
        help='OCR engines to use'
    )
    
    parser.add_argument(
        '--preprocessing', '-p',
        choices=['basic', 'comprehensive', 'aggressive'],
        default='comprehensive',
        help='Image preprocessing method'
    )
    
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Skip saving results to database'
    )
    
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Skip exporting results to files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(
            engines=args.engines,
            preprocessing_method=args.preprocessing
        )
        
        # Process batch
        results = processor.process_batch(
            input_dir=args.input_dir,
            save_to_db=not args.no_db,
            export_results=not args.no_export,
            output_dir=args.output_dir
        )
        
        # Print summary
        print(f"\n{'='*50}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Total images: {results['total_images']}")
        print(f"Processed: {results['processed']}")
        print(f"Failed: {results['failed']}")
        print(f"Success rate: {(results['processed']/results['total_images']*100):.2f}%")
        print(f"Total time: {results['total_time']:.2f}s")
        print(f"Duration: {results['duration']:.2f}s")
        
        if results['failed'] > 0:
            print(f"\nFailed images:")
            for result in results['results']:
                if not result['success']:
                    print(f"  - {result['image_path']}: {result.get('error', 'Unknown error')}")
        
        print(f"\n{'='*50}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
