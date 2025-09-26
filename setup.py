#!/usr/bin/env python3
"""
Setup script for Modern OCR System
=================================

This script helps set up the OCR system by:
1. Installing required dependencies
2. Checking system requirements
3. Creating necessary directories
4. Initializing the database
5. Creating sample images for testing
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print("\n🔍 Checking Tesseract OCR installation...")
    
    try:
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Tesseract found: {version_line}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ Tesseract OCR not found")
    print("\n📋 Installation instructions:")
    
    system = platform.system().lower()
    if system == "darwin":  # macOS
        print("macOS: brew install tesseract")
    elif system == "linux":
        print("Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("CentOS/RHEL: sudo yum install tesseract")
    elif system == "windows":
        print("Windows: Download from https://github.com/tesseract-ocr/tesseract/releases")
    
    print("\nOr visit: https://github.com/tesseract-ocr/tesseract")
    return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "sample_images",
        "logs",
        "temp",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def initialize_database():
    """Initialize the SQLite database"""
    print("\n🗄️ Initializing database...")
    
    try:
        from modern_ocr import OCRDatabase
        db = OCRDatabase()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return False

def create_sample_images():
    """Create sample images for testing"""
    print("\n🖼️ Creating sample images...")
    
    try:
        from modern_ocr import create_sample_images
        create_sample_images()
        print("✅ Sample images created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample images: {e}")
        return False

def test_ocr_system():
    """Test the OCR system with sample images"""
    print("\n🧪 Testing OCR system...")
    
    try:
        from modern_ocr import MultiEngineOCR
        import cv2
        
        ocr = MultiEngineOCR()
        
        # Test with a simple image
        sample_path = "sample_images/clean_text.png"
        if os.path.exists(sample_path):
            image = cv2.imread(sample_path)
            results = ocr.extract_text_all_engines(image)
            
            if results['best']['text']:
                print("✅ OCR system working correctly")
                print(f"Sample result: {results['best']['text'][:50]}...")
                return True
            else:
                print("⚠️ OCR system initialized but no text detected")
                return True
        else:
            print("⚠️ Sample image not found, skipping test")
            return True
            
    except Exception as e:
        print(f"❌ OCR system test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Modern OCR System Setup")
    print("=" * 40)
    
    success_count = 0
    total_steps = 6
    
    # Check Python version
    if check_python_version():
        success_count += 1
    
    # Install dependencies
    if install_dependencies():
        success_count += 1
    
    # Check Tesseract
    if check_tesseract():
        success_count += 1
    
    # Create directories
    create_directories()
    success_count += 1
    
    # Initialize database
    if initialize_database():
        success_count += 1
    
    # Create sample images
    if create_sample_images():
        success_count += 1
    
    # Test OCR system
    if test_ocr_system():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"📊 Setup Summary: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Run the web interface: streamlit run app.py")
        print("2. Or run the command line version: python modern_ocr.py")
        print("3. Upload images and start extracting text!")
    else:
        print("⚠️ Setup completed with some issues")
        print("Please check the error messages above and resolve them")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
