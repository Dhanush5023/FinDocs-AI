import io
import time
import uuid
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Dict, Any, Optional
import pytesseract
from app.core.schemas import ExtractedPage, ExtractedTable, ExtractionResult, DocumentChunk

class DocumentExtractor:
    """
    Production-grade document extraction engine for financial documents.
    Handles digital PDFs, scanned PDF pages via OCR, and raw invoice images.
    """
    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _preprocess_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Applies grayscale, contrast boost, and slight thresholding for financial text clarity."""
        gray = img.convert("L")
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)
        return enhanced

    def _table_to_markdown(self, table_data: List[List[Optional[str]]]) -> str:
        """Converts raw 2D table grid into a clean, LLM-optimized Markdown representation."""
        if not table_data or len(table_data) == 0:
            return ""

        clean_rows = []
        for row in table_data:
            clean_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
            if any(cell for cell in clean_row):  # skip completely empty rows
                clean_rows.append(clean_row)

        if not clean_rows:
            return ""

        headers = clean_rows[0]
        # Pad columns
        max_cols = max(len(r) for r in clean_rows)
        headers += [""] * (max_cols - len(headers))
        
        md_lines = []
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        for row in clean_rows[1:]:
            row += [""] * (max_cols - len(row))
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    def extract_from_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> List[ExtractedPage]:
        """Extracts text & tables from PDF with automatic scanned-page OCR fallback."""
        extracted_pages = []
        
        # 1. Open with pdfplumber for high-fidelity table extraction
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_pdf:
            doc_fitz = fitz.open(stream=pdf_bytes, filetype="pdf")

            for page_idx, plumber_page in enumerate(plumber_pdf.pages):
                page_num = page_idx + 1
                page_tables: List[ExtractedTable] = []
                
                # Extract tables using pdfplumber heuristics
                raw_tables = plumber_page.extract_tables()
                if raw_tables:
                    for t_idx, raw_table in enumerate(raw_tables):
                        md = self._table_to_markdown(raw_table)
                        if md:
                            headers = [str(c).strip() for c in raw_table[0] if c]
                            page_tables.append(ExtractedTable(
                                table_index=t_idx,
                                page_number=page_num,
                                headers=headers,
                                rows=[[str(c) if c else "" for c in r] for r in raw_table[1:]],
                                markdown_repr=md,
                                row_count=len(raw_table),
                                col_count=len(raw_table[0]) if raw_table else 0
                            ))

                # Extract digital text
                page_text = plumber_page.extract_text() or ""
                is_ocr_fallback = False

                # If text density is suspiciously low (< 40 chars), page is likely a scanned image
                if len(page_text.strip()) < 40 and page_idx < len(doc_fitz):
                    fitz_page = doc_fitz[page_idx]
                    pix = fitz_page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    proc_img = self._preprocess_image_for_ocr(img)
                    try:
                        ocr_text = pytesseract.image_to_string(proc_img)
                        if len(ocr_text.strip()) > len(page_text.strip()):
                            page_text = ocr_text
                            is_ocr_fallback = True
                    except Exception:
                        pass # Fallback gracefully if tesseract binary is not present

                extracted_pages.append(ExtractedPage(
                    page_number=page_num,
                    raw_text=page_text.strip(),
                    tables=page_tables,
                    is_ocr_fallback=is_ocr_fallback,
                    char_count=len(page_text.strip())
                ))

            doc_fitz.close()
            
        return extracted_pages

    def extract_from_image_bytes(self, image_bytes: bytes, filename: str = "receipt.png") -> List[ExtractedPage]:
        """Extracts text from raw receipt or invoice images using OCR."""
        img = Image.open(io.BytesIO(image_bytes))
        proc_img = self._preprocess_image_for_ocr(img)
        
        ocr_text = ""
        try:
            ocr_text = pytesseract.image_to_string(proc_img)
        except Exception as e:
            ocr_text = f"[OCR Extraction Unavailable: {str(e)}]"

        return [ExtractedPage(
            page_number=1,
            raw_text=ocr_text.strip(),
            tables=[],
            is_ocr_fallback=True,
            char_count=len(ocr_text.strip())
        )]
