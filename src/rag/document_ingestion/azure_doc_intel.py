"""Azure Document Intelligence (Form Recognizer) extractor.

Extracts text, tables, and layout from unstructured documents (PDF/DOCX/images)
using Azure Form Recognizer's prebuilt-layout model, producing output compatible
with the existing ``MultiModalDocumentProcessor`` format
(:class:`~rag.document_ingestion.enhanced_file_reader.MultiModalDocumentProcessor`
``content`` dict: ``{text, tables, images, metadata}``).

Typical usage::

    from rag.document_ingestion.azure_doc_intel import AzureDocIntelligenceExtractor
    extractor = AzureDocIntelligenceExtractor(
        endpoint="https://myaccount.formrecognizer.azure.com/",
        key="...",
        model_id="prebuilt-layout",
    )
    result = extractor.extract("/path/to/document.pdf")
    # result['content'] == {'text': '...', 'tables': [...], 'images': [], 'metadata': {...}}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List  # noqa: UP035

logger = logging.getLogger(__name__)


class AzureDocIntelligenceExtractor:
    """Extract document content using Azure Form Recognizer."""

    def __init__(
        self,
        endpoint: str,
        key: str,
        model_id: str = "prebuilt-layout",
    ) -> None:
        self.endpoint = endpoint
        self.key = key
        self.model_id = model_id

        # Lazy import to avoid hard dependency when not configured
        from azure.ai.formrecognizer import FormRecognizerClient
        from azure.core.credentials import AzureKeyCredential

        self._client = FormRecognizerClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def extract(self, file_path: str | Path) -> Dict[str, Any]:
        """Extract content from a single document file.

        Args:
            file_path: Path to the PDF/DOCX file.

        Returns:
            Dict with keys ``file_path`` and ``content`` where ``content`` matches
            the format expected by ``MultiModalDocumentProcessor``:
            ``{text: str, tables: list, images: list, metadata: dict}``.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        # Read file bytes
        with open(path, "rb") as f:
            binary_data = f.read()

        # Call Form Recognizer
        try:
            poller = self._client.begin_analyze_document(
                self.model_id, binary_data, content_type="application/octet-stream"
            )
            result = poller.result()
        except Exception as exc:  # noqa: BLE001 - network / auth error
            logger.error("Azure Document Intelligence extraction failed: %s", exc)
            raise

        return self._transform(result, path)

    def _transform(self, result: Any, path: Path) -> Dict[str, Any]:
        """Convert Form Recognizer response to our internal format."""

        # Basic safeguard: if result has no analyzeResult, return minimal content
        analyze = getattr(result, "analyze_result", None) or getattr(result, "analyzeResult", None)
        if analyze is None:
            logger.warning("No analyzeResult in Form Recognizer response")
            return {
                "file_path": str(path),
                "content": {
                    "text": "",
                    "tables": [],
                    "images": [],
                    "metadata": {
                        "file_name": path.name,
                        "file_size": path.stat().st_size,
                        "pages": 0,
                    },
                },
            }

        # --- Text -----------------------------------------------------------
        pages_text: List[str] = []
        page_count = len(getattr(analyze, "pages", []) or [])
        for page in analyze.pages or []:
            page_text = getattr(page, "content", "") or ""
            pages_text.append(page_text)
        full_text = "\n\n".join(pages_text)

        # --- Tables ---------------------------------------------------------
        tables: List[Dict[str, Any]] = []
        for table in analyze.tables or []:
            # Determine grid dimensions
            max_row = max((c.row_index for c in table.cells), default=-1)
            max_col = max((c.column_index for c in table.cells), default=-1)
            n_rows = max_row + 1
            n_cols = max_col + 1

            # Build 2D matrix initialised to empty strings
            matrix: List[List[str]] = [[""] * n_cols for _ in range(n_rows)]

            # Populate cells
            for cell in table.cells or []:
                r = cell.row_index
                c = cell.column_index
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    matrix[r][c] = cell.content or ""

            # Attempt simple header inference: if the first row contains markedly
            # longer strings than others, treat it as headers (optional heuristic)
            headers: List[str] = []
            if n_rows > 1 and n_cols > 0:
                first_row = matrix[0]
                # Heuristic: if first row has more non-empty cells than other rows,
                # treat it as a header row
                non_empty_first = sum(1 for v in first_row if v.strip())
                non_empty_others = sum(
                    1 for r in matrix[1:] for v in r if v.strip()
                )
                if non_empty_first > non_empty_others * 1.5:
                    headers = [v.strip() or "" for v in first_row]

            tables.append(
                {
                    "table_id": f"fr_table_{table.table_id or 0}",
                    "page": getattr(table, "page_number", 1),
                    "data": matrix,
                    "headers": headers,
                    "source": "azure_doc_intelligence",
                }
            )

        # --- Images ---------------------------------------------------------
        # Form Recognizer does not return image OCR/text metadata; we leave images
        # empty and let the downstream pipeline (PyMuPDF image extraction) fill them in
        # if desired. For now images list is empty.
        images: List[Dict[str, Any]] = []

        # --- Metadata -------------------------------------------------------
        metadata: Dict[str, Any] = {
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "pages": page_count,
        }

        content: Dict[str, Any] = {
            "text": full_text,
            "tables": tables,
            "images": images,
            "metadata": metadata,
        }

        return {"file_path": str(path), "content": content}