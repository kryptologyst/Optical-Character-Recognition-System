"""
Streamlit Web UI for Modern OCR System
=====================================

A beautiful and interactive web interface for the OCR system with:
- Image upload and processing
- Real-time OCR results comparison
- Batch processing capabilities
- Results visualization and statistics
- Database management
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image as PILImage
import io
import base64
from datetime import datetime
import os
from pathlib import Path

# Import our OCR system
from modern_ocr import MultiEngineOCR, OCRDatabase, AdvancedImagePreprocessor

# Page configuration
st.set_page_config(
    page_title="Modern OCR System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .ocr-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ocr_system' not in st.session_state:
    st.session_state.ocr_system = None
if 'database' not in st.session_state:
    st.session_state.database = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []

def initialize_systems():
    """Initialize OCR system and database"""
    if st.session_state.ocr_system is None:
        with st.spinner("Initializing OCR engines..."):
            st.session_state.ocr_system = MultiEngineOCR()
            st.session_state.database = OCRDatabase()
        st.success("OCR systems initialized successfully!")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📄 Modern OCR System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📸 Single Image OCR", "📁 Batch Processing", "📊 Statistics", "🗄️ Database"]
    )
    
    # Initialize systems
    initialize_systems()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📸 Single Image OCR":
        show_single_image_page()
    elif page == "📁 Batch Processing":
        show_batch_processing_page()
    elif page == "📊 Statistics":
        show_statistics_page()
    elif page == "🗄️ Database":
        show_database_page()

def show_home_page():
    """Display home page with system overview"""
    st.markdown("## Welcome to the Modern OCR System!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔧 Features
        - **Multiple OCR Engines**: Tesseract, EasyOCR, PaddleOCR
        - **Advanced Preprocessing**: Noise reduction, deskewing, enhancement
        - **Confidence Scoring**: Automatic best result selection
        - **Batch Processing**: Handle multiple images at once
        - **Database Storage**: Track all OCR results
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Capabilities
        - **Text Detection**: Extract text from images
        - **Multi-language Support**: English, Spanish, French, German, Italian
        - **Image Enhancement**: Automatic preprocessing for better results
        - **Real-time Processing**: Fast OCR with live results
        - **Export Options**: Download results as CSV/JSON
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Performance
        - **High Accuracy**: Multiple engines for comparison
        - **Fast Processing**: Optimized algorithms
        - **Scalable**: Handle large batches efficiently
        - **Reliable**: Robust error handling
        - **User-friendly**: Intuitive web interface
        """)
    
    # Quick start section
    st.markdown("## 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing text to extract"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = PILImage.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Quick OCR button
            if st.button("🔍 Quick OCR", type="primary"):
                with st.spinner("Processing image..."):
                    # Convert PIL to OpenCV format
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Process with OCR
                    results = st.session_state.ocr_system.extract_text_all_engines(img_array)
                    
                    # Display results
                    st.markdown("### OCR Results")
                    
                    for engine, result in results.items():
                        if engine != 'best' and 'error' not in result:
                            with st.expander(f"{engine.upper()} Results"):
                                st.write(f"**Text:** {result.get('text', 'N/A')}")
                                st.write(f"**Confidence:** {result.get('confidence', 0):.2f}%")
                                st.write(f"**Word Count:** {result.get('word_count', 0)}")
                    
                    # Best result
                    best = results['best']
                    st.markdown("### 🏆 Best Result")
                    st.markdown(f"""
                    **Engine:** {best.get('engine', 'N/A')}  
                    **Text:** {best.get('text', 'N/A')}  
                    **Confidence:** {best.get('confidence', 0):.2f}%
                    """)
    
    with col2:
        st.markdown("### System Status")
        
        # Check system status
        ocr_status = "✅ Ready" if st.session_state.ocr_system else "❌ Not Initialized"
        db_status = "✅ Connected" if st.session_state.database else "❌ Not Connected"
        
        st.markdown(f"""
        - **OCR System:** {ocr_status}
        - **Database:** {db_status}
        """)
        
        # Statistics preview
        if st.session_state.database:
            stats = st.session_state.database.get_statistics()
            st.markdown("### 📊 Quick Stats")
            st.metric("Total Processed", stats['total_results'])
            
            avg_conf = stats['average_confidences']
            st.metric("Avg Tesseract Confidence", f"{avg_conf['tesseract']:.1f}%")
            st.metric("Avg EasyOCR Confidence", f"{avg_conf['easyocr']:.1f}%")
            st.metric("Avg PaddleOCR Confidence", f"{avg_conf['paddleocr']:.1f}%")

def show_single_image_page():
    """Single image OCR processing page"""
    st.markdown("## 📸 Single Image OCR Processing")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image for OCR processing",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Select an image containing text to extract"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = PILImage.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            preprocessing_method = st.selectbox(
                "Preprocessing Method",
                ["comprehensive", "basic", "aggressive"],
                help="Choose image preprocessing method"
            )
        
        with col2:
            engines_to_use = st.multiselect(
                "OCR Engines",
                ["tesseract", "easyocr", "paddleocr"],
                default=["tesseract", "easyocr", "paddleocr"],
                help="Select which OCR engines to use"
            )
        
        # Process button
        if st.button("🔍 Process Image", type="primary"):
            with st.spinner("Processing image with OCR engines..."):
                # Convert PIL to OpenCV format
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Process with selected engines
                results = {}
                processing_times = {}
                
                start_time = datetime.now()
                
                if "tesseract" in engines_to_use:
                    t_start = datetime.now()
                    results['tesseract'] = st.session_state.ocr_system.extract_text_tesseract(
                        img_array, preprocess=(preprocessing_method != "basic")
                    )
                    processing_times['tesseract'] = (datetime.now() - t_start).total_seconds()
                
                if "easyocr" in engines_to_use:
                    t_start = datetime.now()
                    results['easyocr'] = st.session_state.ocr_system.extract_text_easyocr(img_array)
                    processing_times['easyocr'] = (datetime.now() - t_start).total_seconds()
                
                if "paddleocr" in engines_to_use:
                    t_start = datetime.now()
                    results['paddleocr'] = st.session_state.ocr_system.extract_text_paddleocr(img_array)
                    processing_times['paddleocr'] = (datetime.now() - t_start).total_seconds()
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                # Find best result
                best_result = st.session_state.ocr_system._find_best_result(results)
                results['best'] = best_result
                
                # Display results
                st.markdown("## 📋 OCR Results")
                
                # Results tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🏆 Best Result", "📊 All Results", "⏱️ Performance", "💾 Save Results"])
                
                with tab1:
                    st.markdown("### 🏆 Best OCR Result")
                    st.markdown(f"""
                    **Engine:** {best_result.get('engine', 'N/A')}  
                    **Text:** {best_result.get('text', 'N/A')}  
                    **Confidence:** {best_result.get('confidence', 0):.2f}%  
                    **Word Count:** {best_result.get('word_count', 0)}  
                    **Character Count:** {best_result.get('char_count', 0)}
                    """)
                
                with tab2:
                    st.markdown("### 📊 All Engine Results")
                    
                    for engine, result in results.items():
                        if engine != 'best':
                            with st.expander(f"{engine.upper()} Results"):
                                if 'error' in result:
                                    st.error(f"Error: {result['error']}")
                                else:
                                    st.write(f"**Text:** {result.get('text', 'N/A')}")
                                    st.write(f"**Confidence:** {result.get('confidence', 0):.2f}%")
                                    st.write(f"**Word Count:** {result.get('word_count', 0)}")
                                    st.write(f"**Character Count:** {result.get('char_count', 0)}")
                                    if 'detections' in result:
                                        st.write(f"**Detections:** {result.get('detections', 0)}")
                
                with tab3:
                    st.markdown("### ⏱️ Processing Performance")
                    
                    # Performance metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Processing Time", f"{total_time:.2f}s")
                    
                    with col2:
                        avg_time = sum(processing_times.values()) / len(processing_times) if processing_times else 0
                        st.metric("Average Engine Time", f"{avg_time:.2f}s")
                    
                    with col3:
                        st.metric("Images Processed", "1")
                    
                    # Performance chart
                    if processing_times:
                        df_perf = pd.DataFrame(list(processing_times.items()), columns=['Engine', 'Time (s)'])
                        fig = px.bar(df_perf, x='Engine', y='Time (s)', title="Processing Time by Engine")
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.markdown("### 💾 Save Results to Database")
                    
                    if st.button("💾 Save to Database"):
                        metadata = {
                            'image_hash': str(hash(uploaded_file.name)),
                            'preprocessing_method': preprocessing_method,
                            'processing_time': total_time,
                            'image_width': image.width,
                            'image_height': image.height
                        }
                        
                        result_id = st.session_state.database.save_result(
                            uploaded_file.name, results, metadata
                        )
                        
                        st.success(f"Results saved to database with ID: {result_id}")
                        
                        # Add to session state
                        st.session_state.processing_results.append({
                            'id': result_id,
                            'image_name': uploaded_file.name,
                            'best_text': best_result.get('text', ''),
                            'best_engine': best_result.get('engine', ''),
                            'best_confidence': best_result.get('confidence', 0),
                            'timestamp': datetime.now()
                        })

def show_batch_processing_page():
    """Batch processing page"""
    st.markdown("## 📁 Batch Processing")
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Upload multiple images for batch processing",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Select multiple images to process in batch"
    )
    
    if uploaded_files:
        st.markdown(f"### 📊 Processing {len(uploaded_files)} Images")
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            preprocessing_method = st.selectbox(
                "Preprocessing Method",
                ["comprehensive", "basic", "aggressive"],
                key="batch_preprocessing"
            )
        
        with col2:
            engines_to_use = st.multiselect(
                "OCR Engines",
                ["tesseract", "easyocr", "paddleocr"],
                default=["tesseract", "easyocr", "paddleocr"],
                key="batch_engines"
            )
        
        # Process batch button
        if st.button("🚀 Process Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            batch_results = []
            successful = 0
            failed = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Convert to OpenCV format
                    image = PILImage.open(uploaded_file)
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Process with OCR
                    results = {}
                    
                    if "tesseract" in engines_to_use:
                        results['tesseract'] = st.session_state.ocr_system.extract_text_tesseract(
                            img_array, preprocess=(preprocessing_method != "basic")
                        )
                    
                    if "easyocr" in engines_to_use:
                        results['easyocr'] = st.session_state.ocr_system.extract_text_easyocr(img_array)
                    
                    if "paddleocr" in engines_to_use:
                        results['paddleocr'] = st.session_state.ocr_system.extract_text_paddleocr(img_array)
                    
                    # Find best result
                    best_result = st.session_state.ocr_system._find_best_result(results)
                    results['best'] = best_result
                    
                    batch_results.append({
                        'filename': uploaded_file.name,
                        'success': True,
                        'best_text': best_result.get('text', ''),
                        'best_engine': best_result.get('engine', ''),
                        'best_confidence': best_result.get('confidence', 0),
                        'results': results
                    })
                    
                    successful += 1
                    
                except Exception as e:
                    batch_results.append({
                        'filename': uploaded_file.name,
                        'success': False,
                        'error': str(e),
                        'best_text': '',
                        'best_engine': '',
                        'best_confidence': 0
                    })
                    failed += 1
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Batch processing completed!")
            
            # Display results
            st.markdown("### 📋 Batch Processing Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", len(uploaded_files))
            with col2:
                st.metric("Successful", successful)
            with col3:
                st.metric("Failed", failed)
            
            # Results table
            df_results = pd.DataFrame([
                {
                    'Filename': result['filename'],
                    'Status': '✅ Success' if result['success'] else '❌ Failed',
                    'Best Engine': result['best_engine'],
                    'Confidence': f"{result['best_confidence']:.2f}%" if result['success'] else 'N/A',
                    'Text Preview': result['best_text'][:50] + '...' if len(result['best_text']) > 50 else result['best_text']
                }
                for result in batch_results
            ])
            
            st.dataframe(df_results, use_container_width=True)
            
            # Download results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"batch_ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def show_statistics_page():
    """Statistics and analytics page"""
    st.markdown("## 📊 OCR Statistics & Analytics")
    
    if not st.session_state.database:
        st.error("Database not initialized")
        return
    
    # Get statistics
    stats = st.session_state.database.get_statistics()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images Processed", stats['total_results'])
    
    with col2:
        avg_conf = stats['average_confidences']
        overall_avg = (avg_conf['tesseract'] + avg_conf['easyocr'] + avg_conf['paddleocr']) / 3
        st.metric("Overall Avg Confidence", f"{overall_avg:.1f}%")
    
    with col3:
        best_engine = max(stats['engine_distribution'], key=stats['engine_distribution'].get) if stats['engine_distribution'] else 'N/A'
        st.metric("Most Used Engine", best_engine)
    
    with col4:
        total_engines = sum(stats['engine_distribution'].values())
        st.metric("Total Engine Uses", total_engines)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Engine distribution pie chart
        if stats['engine_distribution']:
            fig_pie = px.pie(
                values=list(stats['engine_distribution'].values()),
                names=list(stats['engine_distribution'].keys()),
                title="Engine Usage Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence comparison bar chart
        engines = list(avg_conf.keys())
        confidences = list(avg_conf.values())
        
        fig_bar = px.bar(
            x=engines,
            y=confidences,
            title="Average Confidence by Engine",
            labels={'x': 'OCR Engine', 'y': 'Average Confidence (%)'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Recent results table
    st.markdown("### 📋 Recent OCR Results")
    
    df_recent = st.session_state.database.get_results(limit=20)
    
    if not df_recent.empty:
        # Select columns to display
        display_columns = ['timestamp', 'image_path', 'best_engine', 'best_confidence', 'best_text']
        available_columns = [col for col in display_columns if col in df_recent.columns]
        
        if available_columns:
            df_display = df_recent[available_columns].copy()
            df_display['best_text'] = df_display['best_text'].apply(lambda x: x[:100] + '...' if len(str(x)) > 100 else x)
            
            st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No OCR results found in database")

def show_database_page():
    """Database management page"""
    st.markdown("## 🗄️ Database Management")
    
    if not st.session_state.database:
        st.error("Database not initialized")
        return
    
    # Database operations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Database Statistics")
        stats = st.session_state.database.get_statistics()
        
        st.metric("Total Records", stats['total_results'])
        st.metric("Database Size", f"{os.path.getsize('ocr_results.db') / 1024:.1f} KB")
        
        # Engine distribution
        st.markdown("#### Engine Distribution")
        for engine, count in stats['engine_distribution'].items():
            st.write(f"**{engine}:** {count} records")
    
    with col2:
        st.markdown("### 🔧 Database Operations")
        
        # Export data
        if st.button("📥 Export All Data"):
            df_all = st.session_state.database.get_results(limit=10000)
            
            csv = df_all.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ocr_database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Clear old records
        st.markdown("#### 🗑️ Data Management")
        
        days_to_keep = st.number_input("Keep records newer than (days)", min_value=1, value=30)
        
        if st.button("🧹 Clean Old Records"):
            # This would need to be implemented in the database class
            st.info("Clean old records functionality would be implemented here")
    
    # Search functionality
    st.markdown("### 🔍 Search Results")
    
    search_term = st.text_input("Search in extracted text:")
    
    if search_term:
        # This would need to be implemented in the database class
        st.info("Search functionality would be implemented here")
    
    # Raw data view
    st.markdown("### 📋 Raw Data View")
    
    limit = st.slider("Number of records to display", min_value=10, max_value=1000, value=50)
    
    df_raw = st.session_state.database.get_results(limit=limit)
    
    if not df_raw.empty:
        st.dataframe(df_raw, use_container_width=True)
    else:
        st.info("No data available")

if __name__ == "__main__":
    main()
