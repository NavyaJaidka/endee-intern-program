"""
Document Ingestion Module for AI Research & Code Copilot.

This module handles document processing for various file formats:
- PDF files (.pdf)
- Text files (.txt, .md, .json)
- GitHub repository files

It provides smart text chunking with overlap for optimal RAG performance.
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import base64

from backend.core.logging import get_logger
from backend.core.config import settings

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """
    Represents a chunk of text from a document.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        text: Text content
        source: Source file or URL
        start_char: Starting character position
        end_char: Ending character position
        metadata: Additional metadata
    """
    chunk_id: str
    text: str
    source: str
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata
        }


@dataclass
class Document:
    """
    Represents a document with metadata.
    
    Attributes:
        doc_id: Unique identifier
        title: Document title
        content: Full text content
        source: Source path or URL
        file_type: File extension
        metadata: Additional metadata
        chunks: List of text chunks
    """
    doc_id: str
    title: str
    content: str
    source: str
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }


class DocumentProcessor:
    """
    Main document processor for handling various file formats.
    
    Supports:
    - PDF files via PyPDF2
    - Text files (txt, md, json)
    - GitHub repository files
    
    Provides smart chunking with configurable overlap.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        logger.info(
            f"DocumentProcessor initialized with "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )
    
    def generate_id(self, text: str) -> str:
        """
        Generate a unique ID from text.
        
        Args:
            text: Input text
        
        Returns:
            MD5 hash of text
        """
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    # ==================== PDF Processing ====================
    
    def process_pdf(self, file_path: str) -> Document:
        """
        Process a PDF file.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Document object
        """
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            logger.error("PyPDF2 not installed. Run: pip install PyPDF2")
            raise ImportError("PyPDF2 is required for PDF processing")
        
        reader = PdfReader(file_path)
        
        # Extract text from all pages
        full_text = ""
        page_count = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
            logger.debug(f"Extracted page {i+1}/{page_count}")
        
        # Get title from filename
        title = Path(file_path).stem
        
        # Create document
        doc = Document(
            doc_id=self.generate_id(file_path),
            title=title,
            content=full_text,
            source=file_path,
            file_type="pdf",
            metadata={
                "page_count": page_count,
                "file_size": os.path.getsize(file_path)
            }
        )
        
        # Generate chunks
        doc.chunks = self.chunk_text(full_text, source=file_path)
        
        logger.info(f"Processed PDF with {len(doc.chunks)} chunks")
        
        return doc
    
    # ==================== Text Processing ====================
    
    def process_txt(self, file_path: str) -> Document:
        """
        Process a text file.
        
        Args:
            file_path: Path to text file
        
        Returns:
            Document object
        """
        logger.info(f"Processing text file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        title = Path(file_path).stem
        file_ext = Path(file_path).suffix.lower()
        
        doc = Document(
            doc_id=self.generate_id(file_path),
            title=title,
            content=content,
            source=file_path,
            file_type=file_ext.replace(".", ""),
            metadata={
                "file_size": os.path.getsize(file_path),
                "line_count": len(content.split('\n'))
            }
        )
        
        doc.chunks = self.chunk_text(content, source=file_path)
        
        logger.info(f"Processed text file with {len(doc.chunks)} chunks")
        
        return doc
    
    def process_markdown(self, file_path: str) -> Document:
        """
        Process a markdown file.
        
        Args:
            file_path: Path to markdown file
        
        Returns:
            Document object
        """
        # Process as text first
        doc = self.process_txt(file_path)
        doc.file_type = "md"
        
        # Extract markdown headers as metadata
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find headers
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        doc.metadata["headers"] = headers[:10]  # Limit to first 10 headers
        
        return doc
    
    def process_json(self, file_path: str) -> Document:
        """
        Process a JSON file.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Document object
        """
        logger.info(f"Processing JSON file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to text representation
        content = json.dumps(data, indent=2)
        
        title = Path(file_path).stem
        
        doc = Document(
            doc_id=self.generate_id(file_path),
            title=title,
            content=content,
            source=file_path,
            file_type="json",
            metadata={
                "file_size": os.path.getsize(file_path),
                "json_keys": list(data.keys()) if isinstance(data, dict) else None
            }
        )
        
        doc.chunks = self.chunk_text(content, source=file_path)
        
        return doc
    
    # ==================== GitHub Processing ====================
    
    def process_github_repo(
        self,
        repo_url: str,
        branch: str = "main"
    ) -> List[Document]:
        """
        Process files from a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
            branch: Branch to fetch
        
        Returns:
            List of Document objects
        """
        logger.info(f"Processing GitHub repo: {repo_url}")
        
        try:
            import git
        except ImportError:
            logger.error("GitPython not installed. Run: pip install gitpython")
            raise ImportError("GitPython is required for GitHub processing")
        
        # Parse repo URL
        # Expected format: https://github.com/owner/repo
        match = re.match(r'github\.com/([^/]+)/([^/]+)', repo_url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        
        owner, repo = match.groups()
        repo_name = repo.replace('.git', '')
        
        # Clone or pull the repository
        clone_dir = Path(f"/tmp/github_{owner}_{repo_name}")
        
        if clone_dir.exists():
            logger.info(f"Repository already exists, pulling latest")
            try:
                repo_obj = git.Repo(clone_dir)
                origin = repo_obj.remotes.origin
                origin.pull()
            except Exception as e:
                logger.warning(f"Pull failed, removing and re-cloning: {e}")
                import shutil
                shutil.rmtree(clone_dir)
                clone_dir = None
        
        if not clone_dir.exists():
            logger.info(f"Cloning repository: {owner}/{repo_name}")
            git.Git().clone(
                repo_url,
                clone_dir,
                branch=branch,
                depth=1  # Shallow clone for efficiency
            )
        
        # Process all files
        documents = []
        supported_extensions = {'.py', '.js', '.ts', '.txt', '.md', '.json', '.yaml', '.yml', '.java', '.cpp', '.c', '.go', '.rs'}
        
        for file_path in clone_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in supported_extensions:
                # Skip test files and node_modules
                if 'test' in str(file_path) or 'node_modules' in str(file_path):
                    continue
                
                try:
                    rel_path = file_path.relative_to(clone_dir)
                    doc = self.process_file(str(file_path))
                    doc.metadata["github_repo"] = f"{owner}/{repo_name}"
                    doc.metadata["github_path"] = str(rel_path)
                    doc.source = f"github://{owner}/{repo_name}/{rel_path}"
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Processed {len(documents)} files from GitHub repo")
        
        return documents
    
    def fetch_github_file(self, repo_url: str, file_path: str) -> str:
        """
        Fetch a single file from GitHub without cloning.
        
        Args:
            repo_url: GitHub repository URL
            file_path: Path to file in repo
        
        Returns:
            File content as string
        """
        import requests
        
        # Parse owner/repo from URL
        match = re.match(r'github\.com/([^/]+)/([^/]+)', repo_url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        
        owner, repo = match.groups()
        repo = repo.replace('.git', '')
        
        # Try raw GitHub first
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}"
        
        headers = {}
        if settings.github_token:
            headers["Authorization"] = f"token {settings.github_token}"
        
        response = requests.get(raw_url, headers=headers)
        
        if response.status_code == 404:
            # Try with master branch
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}"
            response = requests.get(raw_url, headers=headers)
        
        response.raise_for_status()
        
        return response.text
    
    # ==================== Universal Processing ====================
    
    def process_file(self, file_path: str) -> Document:
        """
        Process a file based on its extension.
        
        Args:
            file_path: Path to file
        
        Returns:
            Document object
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return self.process_pdf(file_path)
        elif extension in ['.txt', '.text']:
            return self.process_txt(file_path)
        elif extension in ['.md', '.markdown']:
            return self.process_markdown(file_path)
        elif extension == '.json':
            return self.process_json(file_path)
        else:
            # Try as text file
            return self.process_txt(file_path)
    
    def process_content(
        self,
        content: str,
        title: str,
        source: str = "direct_input"
    ) -> Document:
        """
        Process raw text content.
        
        Args:
            content: Text content
            title: Document title
            source: Source identifier
        
        Returns:
            Document object
        """
        doc = Document(
            doc_id=self.generate_id(content[:100]),
            title=title,
            content=content,
            source=source,
            file_type="txt",
            metadata={"content_type": "direct_input"}
        )
        
        doc.chunks = self.chunk_text(content, source=source)
        
        return doc
    
    # ==================== Text Chunking ====================
    
    def chunk_text(
        self,
        text: str,
        source: str = "unknown"
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Uses a smart chunking strategy:
        1. Split by paragraphs first
        2. Combine small paragraphs into chunks
        3. Ensure chunks don't exceed chunk_size
        4. Add overlap between chunks for context continuity
        
        Args:
            text: Input text
            source: Source identifier
        
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If single paragraph exceeds chunk_size, split it
            if para_size > self.chunk_size:
                # First, save current chunk if not empty
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunk = self._create_chunk(
                        chunk_text,
                        source,
                        chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph
                sub_chunks = self._split_large_paragraph(para)
                for sub_chunk in sub_chunks:
                    chunk = self._create_chunk(sub_chunk, source, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
                
                continue
            
            # Check if adding this paragraph would exceed chunk_size
            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunk = self._create_chunk(
                    chunk_text,
                    source,
                    chunk_index,
                    overlap_text=" ".join(current_chunk[-2:]) if len(current_chunk) > 1 else ""
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_words = " ".join(current_chunk[-2:]).split()
                    overlap_size = min(len(" ".join(overlap_words)), self.chunk_overlap)
                    overlap_start = " ".join(current_chunk)[-overlap_size:]
                    current_chunk = [overlap_start, para] if overlap_start else [para]
                    current_size = len(overlap_start) + para_size if overlap_start else para_size
                else:
                    current_chunk = [para]
                    current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = self._create_chunk(chunk_text, source, chunk_index)
            chunks.append(chunk)
        
        logger.debug(f"Created {len(chunks)} chunks from text")
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines or single newlines with significant spacing
        paragraphs = re.split(r'\n\n+|\n(?=\n)', text)
        
        # Filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _split_large_paragraph(self, text: str) -> List[str]:
        """Split a large paragraph into smaller chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        source: str,
        index: int,
        overlap_text: str = ""
    ) -> DocumentChunk:
        """Create a DocumentChunk object."""
        chunk_id = f"{self.generate_id(source)}_{index}"
        
        # Calculate position
        start_char = index * (self.chunk_size - self.chunk_overlap)
        end_char = start_char + len(text)
        
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            source=source,
            start_char=start_char,
            end_char=end_char,
            metadata={
                "chunk_index": index,
                "overlap": bool(overlap_text)
            }
        )


# Convenience functions
def process_file(file_path: str) -> Document:
    """Process a file and return Document."""
    processor = DocumentProcessor()
    return processor.process_file(file_path)


def process_pdf(file_path: str) -> Document:
    """Process a PDF file."""
    processor = DocumentProcessor()
    return processor.process_pdf(file_path)


def process_text(text: str, title: str) -> Document:
    """Process raw text content."""
    processor = DocumentProcessor()
    return processor.process_content(text, title)


def chunk_text(text: str) -> List[DocumentChunk]:
    """Chunk text into smaller pieces."""
    processor = DocumentProcessor()
    return processor.chunk_text(text)
