"""
Configuration file for Modern OCR System
=======================================

This file contains all configurable parameters for the OCR system.
Modify these settings to customize the behavior of the system.
"""

# OCR Engine Configuration
OCR_CONFIG = {
    # Tesseract settings
    'tesseract': {
        'config': '--oem 3 --psm 6',
        'languages': ['eng'],
        'preprocess': True
    },
    
    # EasyOCR settings
    'easyocr': {
        'languages': ['en', 'es', 'fr', 'de', 'it'],
        'gpu': True,  # Set to False if no GPU available
        'confidence_threshold': 0.5
    },
    
    # PaddleOCR settings
    'paddleocr': {
        'use_angle_cls': True,
        'lang': 'en',
        'use_gpu': True,  # Set to False if no GPU available
        'confidence_threshold': 0.5
    }
}

# Image Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'methods': {
        'basic': {
            'grayscale': True,
            'threshold': True,
            'threshold_method': 'OTSU'
        },
        'comprehensive': {
            'grayscale': True,
            'denoise': True,
            'enhance_contrast': True,
            'deskew': True,
            'threshold': True,
            'threshold_method': 'OTSU'
        },
        'aggressive': {
            'grayscale': True,
            'denoise': True,
            'denoise_iterations': 2,
            'morphology': True,
            'adaptive_threshold': True,
            'threshold_block_size': 11,
            'threshold_c': 2
        }
    },
    
    # CLAHE settings for contrast enhancement
    'clahe': {
        'clip_limit': 2.0,
        'tile_grid_size': (8, 8)
    },
    
    # Deskewing settings
    'deskew': {
        'min_angle_threshold': 0.5,
        'max_angle': 45
    }
}

# Database Configuration
DATABASE_CONFIG = {
    'path': 'ocr_results.db',
    'backup_interval': 24,  # hours
    'max_records': 10000,
    'cleanup_days': 30
}

# Web Interface Configuration
WEB_CONFIG = {
    'title': 'Modern OCR System',
    'page_icon': '📄',
    'layout': 'wide',
    'theme': 'light',
    'max_file_size': 200,  # MB
    'allowed_extensions': ['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    'batch_size': 50
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_workers': 4,  # For batch processing
    'timeout': 300,  # seconds
    'memory_limit': 2048,  # MB
    'cache_size': 100  # Number of images to cache
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/ocr_system.log',
    'max_size': 10,  # MB
    'backup_count': 5
}

# Export Configuration
EXPORT_CONFIG = {
    'csv_encoding': 'utf-8',
    'json_indent': 2,
    'include_images': False,  # Whether to include image data in exports
    'compression': True
}

# Security Configuration
SECURITY_CONFIG = {
    'max_file_size': 200 * 1024 * 1024,  # 200MB in bytes
    'allowed_mime_types': [
        'image/png',
        'image/jpeg',
        'image/jpg',
        'image/bmp',
        'image/tiff'
    ],
    'scan_uploads': True  # Whether to scan uploaded files
}

# Default Settings
DEFAULT_SETTINGS = {
    'preprocessing_method': 'comprehensive',
    'engines': ['tesseract', 'easyocr', 'paddleocr'],
    'confidence_threshold': 50,  # percentage
    'auto_save': True,
    'show_confidence': True,
    'show_processing_time': True
}

# Language Support
LANGUAGE_SUPPORT = {
    'tesseract': [
        'eng', 'spa', 'fra', 'deu', 'ita', 'por', 'rus', 'jpn', 'kor', 'chi_sim'
    ],
    'easyocr': [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'
    ],
    'paddleocr': [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'ch'
    ]
}

# Error Messages
ERROR_MESSAGES = {
    'file_too_large': 'File size exceeds maximum allowed size',
    'unsupported_format': 'Unsupported image format',
    'ocr_failed': 'OCR processing failed',
    'database_error': 'Database operation failed',
    'preprocessing_error': 'Image preprocessing failed',
    'engine_not_available': 'OCR engine not available'
}

# Success Messages
SUCCESS_MESSAGES = {
    'ocr_completed': 'OCR processing completed successfully',
    'batch_completed': 'Batch processing completed',
    'database_saved': 'Results saved to database',
    'export_completed': 'Export completed successfully'
}
