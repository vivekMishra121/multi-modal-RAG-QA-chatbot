"""Example: Chart Detection and Metadata Extraction"""

import logging
from pathlib import Path
from document_ingestion.enhanced_file_reader import MultiModalDocumentProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    pdf_path = Path(r"C:\Users\HP\Downloads\Qatar Test Document.pdf")
    
    print("\n" + "="*70)
    print("CHART DETECTION & METADATA EXTRACTION")
    print("="*70)
    
    processor = MultiModalDocumentProcessor()
    results = processor.process_documents([pdf_path])
    
    if results and 'content' in results[0]:
        content = results[0]['content']
        images = content.get('images', [])
        
        print(f"\nTotal images extracted: {len(images)}")
        
        # Analyze charts
        charts = [img for img in images if img.get('is_chart')]
        regular_images = [img for img in images if not img.get('is_chart')]
        
        print(f"Charts detected: {len(charts)}")
        print(f"Regular images: {len(regular_images)}")
        
        if charts:
            print("\n" + "="*70)
            print("CHART DETAILS")
            print("="*70)
            
            for i, chart in enumerate(charts, 1):
                print(f"\n[Chart {i}] {chart['image_id']}")
                print(f"  Page: {chart['page']}")
                print(f"  Type: {chart.get('chart_type', 'unknown').replace('_', ' ').title()}")
                print(f"  Size: {chart['size']}")
                
                elements = chart.get('chart_elements', {})
                if elements:
                    print(f"  Detected Elements:")
                    if 'bars_detected' in elements:
                        print(f"    - Bars: {elements['bars_detected']}")
                    if 'horizontal_lines' in elements:
                        print(f"    - Horizontal lines: {elements['horizontal_lines']}")
                    if 'vertical_lines' in elements:
                        print(f"    - Vertical lines: {elements['vertical_lines']}")
                    if 'circular_elements' in elements:
                        print(f"    - Pie segments: {elements['circular_elements']}")
                    if 'unique_colors' in elements:
                        print(f"    - Color regions: {elements['unique_colors']}")
                    if 'confidence' in elements:
                        print(f"    - Confidence: {elements['confidence']:.2%}")
                
                ocr_text = chart.get('ocr_text', '').strip()
                if ocr_text:
                    print(f"  OCR Text: {ocr_text[:150]}...")
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Processed {content['metadata']['pages']} pages")
        print(f"✓ Detected {len(charts)} charts with metadata")
        print(f"✓ Chart types found: {set(c.get('chart_type') for c in charts if c.get('chart_type'))}")
        print("="*70)


if __name__ == "__main__":
    main()

