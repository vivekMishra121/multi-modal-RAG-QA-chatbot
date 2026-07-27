"""Concrete chunker implementations for multi-modal content"""

import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import ChunkerInterface, Chunk, ChunkType


class TextChunker(ChunkerInterface):
    """Chunks text content using semantic splitting with page tracking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into semantic chunks with page tracking"""
        if not content or not content.strip():
            return []
        
        # Split by page markers first
        page_sections = self._split_by_pages(content)
        
        all_chunks = []
        for page_num, page_text in page_sections:
            # Chunk each page separately
            text_chunks = self.splitter.split_text(page_text)
            
            for i, text in enumerate(text_chunks):
                # Add context to chunk
                enhanced_text = self._add_context(text, page_num, metadata)
                
                all_chunks.append(
                    Chunk(
                        content=enhanced_text,
                        chunk_type=ChunkType.TEXT,
                        metadata={
                            **metadata, 
                            'page': page_num,
                            'chunk_index': len(all_chunks)
                        },
                        chunk_id=self._generate_id(text, metadata, len(all_chunks))
                    )
                )
        
        return all_chunks
    
    def _split_by_pages(self, content: str) -> List[tuple]:
        """Split content by page markers"""
        page_sections = []
        current_page = 1
        current_text = []
        
        for line in content.split('\n'):
            # Check for page marker
            if line.strip().startswith('## Page'):
                # Save previous page
                if current_text:
                    page_sections.append((current_page, '\n'.join(current_text)))
                    current_text = []
                
                # Extract page number
                try:
                    current_page = int(line.replace('## Page', '').strip())
                except:
                    current_page += 1
            else:
                current_text.append(line)
        
        # Add last page
        if current_text:
            page_sections.append((current_page, '\n'.join(current_text)))
        
        return page_sections
    
    def _add_context(self, text: str, page_num: int, metadata: Dict[str, Any]) -> str:
        """Add document context to chunk"""
        doc_name = metadata.get('file_name', 'Document')
        
        # Clean text (remove extra whitespace)
        text = ' '.join(text.split())
        
        # Add context header
        context_header = f"[Document: {doc_name} | Page: {page_num}]\n"
        
        return context_header + text
    
    @staticmethod
    def _generate_id(content: str, metadata: Dict[str, Any], index: int) -> str:
        """Generate unique chunk ID"""
        source = metadata.get('source', 'unknown')
        hash_input = f"{source}_{index}_{content[:50]}"
        return hashlib.md5(hash_input.encode()).hexdigest()


class TableChunker(ChunkerInterface):
    """Chunks table content with structure preservation"""
    
    def chunk(self, content: Dict[str, Any], metadata: Dict[str, Any]) -> List[Chunk]:
        """Convert table to text representation"""
        if not content or 'data' not in content:
            return []
        
        table_text = self._table_to_text(content)
        
        return [
            Chunk(
                content=table_text,
                chunk_type=ChunkType.TABLE,
                metadata={
                    **metadata,
                    'table_id': content.get('table_id'),
                    'source_method': content.get('source'),
                    'accuracy': content.get('accuracy')
                },
                chunk_id=self._generate_id(content, metadata)
            )
        ]
    
    @staticmethod
    def _table_to_text(table: Dict[str, Any]) -> str:
        """Convert table data to readable text with context"""
        lines = []
        
        # Add table identifier with context
        table_id = table.get('table_id', 'unknown')
        page = table.get('page', 'N/A')
        lines.append(f"[Table: {table_id} | Page: {page}]")
        lines.append("")
        
        # Add headers if available
        headers = table.get('headers', [])
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * 50)
        
        # Add data rows
        data = table.get('data', [])
        for row in data:
            lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)
    
    @staticmethod
    def _generate_id(content: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Generate unique table chunk ID"""
        source = metadata.get('source', 'unknown')
        table_id = content.get('table_id', 'unknown')
        hash_input = f"{source}_{table_id}"
        return hashlib.md5(hash_input.encode()).hexdigest()


class ImageChunker(ChunkerInterface):
    """Chunks image content with OCR text"""
    
    def chunk(self, content: Dict[str, Any], metadata: Dict[str, Any]) -> List[Chunk]:
        """Convert image metadata and OCR to text chunk"""
        if not content or 'ocr_text' not in content:
            return []
        
        ocr_text = content.get('ocr_text', '').strip()
        if not ocr_text or ocr_text in ['[OCR failed]', '[OCR not available]']:
            return []
        
        image_text = self._image_to_text(content)
        
        return [
            Chunk(
                content=image_text,
                chunk_type=ChunkType.IMAGE,
                metadata={
                    **metadata,
                    'image_id': content.get('image_id'),
                    'image_size': content.get('size'),
                    'image_format': content.get('format'),
                    'is_chart': content.get('is_chart', False),
                    'chart_type': content.get('chart_type'),
                    'chart_confidence': content.get('chart_elements', {}).get('confidence', 0.0)
                },
                chunk_id=self._generate_id(content, metadata)
            )
        ]
    
    @staticmethod
    def _image_to_text(image: Dict[str, Any]) -> str:
        """Convert image data to readable text with chart metadata"""
        image_id = image.get('image_id', 'unknown')
        page = image.get('page', 'N/A')
        
        lines = [f"[Image: {image_id} | Page: {page}]"]
        
        # Add chart-specific information
        if image.get('is_chart'):
            chart_type = image.get('chart_type', 'unknown')
            lines.append(f"Type: {chart_type.replace('_', ' ').title()}")
            
            elements = image.get('chart_elements', {})
            if elements:
                if 'bars_detected' in elements:
                    lines.append(f"Bars: {elements['bars_detected']}")
                if 'horizontal_lines' in elements and 'vertical_lines' in elements:
                    lines.append(f"Grid: {elements['horizontal_lines']}x{elements['vertical_lines']} lines")
                if 'circular_elements' in elements:
                    lines.append(f"Segments: {elements['circular_elements']}")
        
        lines.append("")  # Empty line before content
        
        # Add OCR text
        ocr_text = image.get('ocr_text', '').strip()
        if ocr_text:
            lines.append(ocr_text)
        
        return "\n".join(lines)
    
    @staticmethod
    def _generate_id(content: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Generate unique image chunk ID"""
        source = metadata.get('source', 'unknown')
        image_id = content.get('image_id', 'unknown')
        hash_input = f"{source}_{image_id}"
        return hashlib.md5(hash_input.encode()).hexdigest()
