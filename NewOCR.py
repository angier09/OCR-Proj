"""
OCR Pipeline for Arabic/English Text Detection and Recognition
==============================================================
Supports: PDF and Image files (.pdf, .png, .jpg, .jpeg, .tiff, .bmp)
Output: Structured JSON with detected text, language, bounding boxes, and confidence scores

Dependencies:
    pip install pytesseract pdf2image opencv-python-headless Pillow numpy langdetect

System requirements:
    - Tesseract OCR: https://github.com/tesseract-ocr/tesseract
      Linux:   sudo apt install tesseract-ocr tesseract-ocr-ara
      macOS:   brew install tesseract tesseract-lang
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - Poppler (for PDF support):
      Linux:   sudo apt install poppler-utils
      macOS:   brew install poppler
      Windows: https://github.com/oschwartz10612/poppler-windows
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langdetect import detect, DetectorFactory

# Make langdetect deterministic across runs
DetectorFactory.seed = 42

# ─────────────────────────── Logging Setup ───────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — INPUT LOADING
#  Accepts PDF or common image formats and returns a list of PIL Images (one
#  per page / frame) together with some basic file metadata.
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def load_input(file_path: str) -> tuple[list[Image.Image], dict]:
    """
    Load a PDF or image file and return (pages, metadata).

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the input file.

    Returns
    -------
    pages : list[PIL.Image.Image]
        One PIL image per page / frame.
    metadata : dict
        Basic file-level information.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    ext = path.suffix.lower()
    metadata = {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "file_type": "pdf" if ext == ".pdf" else "image",
    }

    if ext == ".pdf":
        logger.info("Loading PDF: %s", path.name)
        # dpi=300 gives a good balance between speed and OCR accuracy
        pages = convert_from_path(str(path), dpi=400)
        metadata["total_pages"] = len(pages)
        logger.info("  → %d page(s) converted to images", len(pages))

    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        logger.info("Loading image: %s", path.name)
        pages = [Image.open(str(path)).convert("RGB")]
        metadata["total_pages"] = 1

    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: .pdf, {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    return pages, metadata


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — IMAGE PRE-PROCESSING
#  Clean and enhance each page image so that Tesseract can extract text more
#  accurately.  The pipeline applies:
#    1. Grayscale conversion   – removes colour noise
#    2. Denoising              – smooths salt-and-pepper artefacts
#    3. Adaptive thresholding  – binarises the image locally (handles uneven
#                                lighting / gradients common in scans)
#    4. Deskewing              – corrects slight rotation caused by scanning
#    5. Upscaling              – ensures small text is large enough for OCR
# ═══════════════════════════════════════════════════════════════════════════════

def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct slight rotation via Hough line transform."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=100, maxLineGap=10,
    )
    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.3 or abs(median_angle) > 10:
        return image

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Enhanced pre-processing pipeline for low-quality scanned Arabic documents.
    Handles watermarks, uneven lighting, broken strokes, and low contrast.

    Returns a binarised OpenCV image ready for Tesseract.
    """

    # ── 0. PIL → OpenCV BGR ──────────────────────────────────────────────────
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # ── 1. Upscale to at least 2400px on the long side ───────────────────────
    #    Arabic script has fine detail (dots, diacritics). More pixels = better OCR.
    h, w = cv_image.shape[:2]
    target = 2400
    long_side = max(h, w)
    if long_side < target:
        scale = target / long_side
        cv_image = cv2.resize(
            cv_image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # ── 2. Grayscale ─────────────────────────────────────────────────────────
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # ── 3. CLAHE (Contrast Limited Adaptive Histogram Equalisation) ──────────
    #    Brings out faded ink and normalises uneven scan lighting.
    #    clipLimit=2.0 prevents over-amplifying noise in blank areas.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ── 4. Watermark suppression via morphological top-hat ───────────────────
    #    Top-hat extracts small bright objects (text) against a slowly varying
    #    background (watermarks, stamps, gradients).
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_tophat)

    # Blend back: keeps text, suppresses large-scale background patterns
    suppressed = cv2.add(enhanced, tophat)

    # ── 5. Strong denoising ──────────────────────────────────────────────────
    #    h=15 is stronger than the default 10 — helps with grainy scans.
    denoised = cv2.fastNlMeansDenoising(suppressed, h=15, templateWindowSize=7, searchWindowSize=21)

    # ── 6. Otsu binarisation ─────────────────────────────────────────────────
    #    Otsu automatically finds the optimal global threshold.
    #    Works better than adaptive thresholding when lighting is now uniform
    #    (after CLAHE). Apply Gaussian blur first to reduce noise impact.
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── 7. Morphological closing ─────────────────────────────────────────────
    #    Arabic letters often have broken strokes in low-res scans.
    #    Closing (dilate → erode) reconnects nearby fragments.
    #    Small kernel (2×2) to avoid merging distinct characters.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # ── 8. Deskew ────────────────────────────────────────────────────────────
    deskewed = _deskew(closed)

    return deskewed

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — LANGUAGE DETECTION
#  Attempts to infer whether the page is primarily Arabic, English, or mixed.
#  This drives the Tesseract language configuration.
# ═══════════════════════════════════════════════════════════════════════════════

def detect_language(image: np.ndarray) -> str:
    """
    Run a fast pass with Tesseract (English only) to grab raw text, then use
    langdetect to determine the dominant language.

    Returns
    -------
    'ara'   – Arabic detected  → use Tesseract's Arabic model
    'eng'   – English detected → use Tesseract's English model
    'ara+eng' – Mixed / uncertain → use both models together
    """
    # Quick single-language pass just to get text for language detection
    raw_text = pytesseract.image_to_string(image, lang="eng+ara", config="--psm 3")
    raw_text = raw_text.strip()

    if not raw_text:
        logger.debug("  No text found in language detection pass → defaulting to eng+ara")
        return "ara+eng"

    try:
        lang_code = detect(raw_text)
    except Exception:
        lang_code = "unknown"

    logger.debug("  langdetect result: %s", lang_code)

    # langdetect returns ISO 639-1 codes; map to Tesseract codes
    if lang_code == "ar":
        return "ara"
    elif lang_code in {"en", "en-US", "en-GB"}:
        return "eng"
    else:
        # For anything else (including mixed or unrecognised) use both
        return "ara+eng"


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — OCR ENGINE WRAPPER
#  Runs Tesseract with the correct language model(s) and extracts:
#    • Full plain text
#    • Per-word bounding boxes + confidence scores  (via image_to_data)
#    • Structured block/paragraph/line hierarchy     (via image_to_data)
# ═══════════════════════════════════════════════════════════════════════════════

# Tesseract PSM 3 = automatic page segmentation with OSD.
# Works well for multi-column layouts (newspapers, reports).
TESSERACT_CONFIG = "--psm 6 --oem 1 -c preserve_interword_spaces=1"


def run_ocr(image: np.ndarray, lang: str) -> dict:
    """
    Perform OCR on a pre-processed image.

    Returns a dict with:
        full_text   : str         – complete page text
        words       : list[dict]  – per-word data (text, bbox, confidence)
        blocks      : list[dict]  – logical text blocks aggregated from words
        language    : str         – Tesseract language string used
    """
    pil_img = Image.fromarray(image)

    # --- Full text ---
    full_text = pytesseract.image_to_string(pil_img, lang=lang, config=TESSERACT_CONFIG)

    # --- Detailed word-level data ---
    data = pytesseract.image_to_data(
        pil_img, lang=lang, config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    n = len(data["text"])
    for i in range(n):
        word_text = data["text"][i].strip()
        if not word_text:
            continue                   # skip empty tokens

        conf = int(data["conf"][i])
        if conf < 0:
            continue                   # Tesseract returns -1 for layout tokens

        words.append({
            "text": word_text,
            "confidence": conf,        # 0-100
            "bounding_box": {
                "x": data["left"][i],
                "y": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            },
            "block_num": data["block_num"][i],
            "par_num":   data["par_num"][i],
            "line_num":  data["line_num"][i],
            "word_num":  data["word_num"][i],
        })

    # --- Aggregate words → lines → paragraphs → blocks ---
    blocks = _aggregate_blocks(words)

    return {
        "full_text": full_text.strip(),
        "words": words,
        "blocks": blocks,
        "language_used": lang,
    }


def _aggregate_blocks(words: list[dict]) -> list[dict]:
    """
    Group words by (block_num, par_num, line_num) to produce a hierarchical
    structure:  blocks → paragraphs → lines → words.
    """
    from collections import defaultdict

    # block_num → par_num → line_num → [word_dict]
    tree: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for w in words:
        tree[w["block_num"]][w["par_num"]][w["line_num"]].append(w)

    blocks = []
    for blk_id, paragraphs in sorted(tree.items()):
        para_list = []
        for par_id, lines in sorted(paragraphs.items()):
            line_list = []
            for ln_id, line_words in sorted(lines.items()):
                line_text = " ".join(w["text"] for w in line_words)
                avg_conf  = round(
                    sum(w["confidence"] for w in line_words) / len(line_words), 1
                )
                # Bounding box that encompasses the whole line
                xs = [w["bounding_box"]["x"] for w in line_words]
                ys = [w["bounding_box"]["y"] for w in line_words]
                x2s = [w["bounding_box"]["x"] + w["bounding_box"]["width"]  for w in line_words]
                y2s = [w["bounding_box"]["y"] + w["bounding_box"]["height"] for w in line_words]
                line_list.append({
                    "line_num": ln_id,
                    "text": line_text,
                    "average_confidence": avg_conf,
                    "bounding_box": {
                        "x": min(xs), "y": min(ys),
                        "width": max(x2s) - min(xs),
                        "height": max(y2s) - min(ys),
                    },
                    "words": line_words,
                })
            para_text = " ".join(ln["text"] for ln in line_list)
            para_list.append({
                "paragraph_num": par_id,
                "text": para_text,
                "lines": line_list,
            })
        block_text = "\n".join(p["text"] for p in para_list)
        blocks.append({
            "block_num": blk_id,
            "text": block_text,
            "paragraphs": para_list,
        })

    return blocks


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — POST-PROCESSING
#  Clean the raw OCR text:  strip control characters, normalise Arabic ligatures,
#  deduplicate whitespace, and compute a page-level confidence score.
# ═══════════════════════════════════════════════════════════════════════════════

import re
import unicodedata


def postprocess_text(ocr_result: dict) -> dict:
    """
    Apply text cleaning and enrichment to an OCR result dict.
    Mutates and returns the same dict with additional keys:
        cleaned_text        : str
        page_confidence     : float  (mean confidence of all words)
        word_count          : int
        character_count     : int
        contains_arabic     : bool
        contains_english    : bool
    """
    raw = ocr_result["full_text"]

    # Remove non-printable / control characters (keep newlines and spaces)
    cleaned = "".join(
        ch for ch in raw
        if unicodedata.category(ch)[0] != "C" or ch in "\n\t "
    )

    # Normalise common Arabic ligatures and presentation forms to base forms
    # U+FE70..U+FEFF are Arabic Presentation Forms-B (often produced by OCR)
    cleaned = _normalise_arabic(cleaned)

    # Collapse multiple spaces / blank lines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    # Page-level statistics
    words = ocr_result["words"]
    page_conf = (
        round(sum(w["confidence"] for w in words) / len(words), 1)
        if words else 0.0
    )

    ocr_result["cleaned_text"]     = cleaned
    ocr_result["page_confidence"]  = page_conf
    ocr_result["word_count"]       = len(words)
    ocr_result["character_count"]  = len(cleaned)
    ocr_result["contains_arabic"]  = bool(re.search(r"[\u0600-\u06FF]", cleaned))
    ocr_result["contains_english"] = bool(re.search(r"[A-Za-z]",         cleaned))

    return ocr_result


def _normalise_arabic(text: str) -> str:
    """
    Map Arabic Presentation Forms (FE70–FEFF) back to their canonical Unicode
    equivalents.  This ensures consistent downstream text handling.
    """
    # unicodedata.normalize with NFKC handles most presentation forms
    return unicodedata.normalize("NFKC", text)

def build_legal_structure(pages_results: list[dict]) -> dict:
    """
    Convert OCR text into structured legal JSON:
    Chapter → Section → Article → Clause

    Adds a 'virtual chapter' if none detected.
    """

    import re

    # Patterns (English + Arabic)
    chapter_pattern = re.compile(r"(chapter|الفصل)\s+(\d+)", re.IGNORECASE)
    section_pattern = re.compile(r"(section|القسم)\s+(\d+)", re.IGNORECASE)
    #article_pattern = re.compile(r"(article|المادة)\s+(\d+)", re.IGNORECASE)

    article_pattern = re.compile(
        r"(article|المادة)\s*[\(\-–—]?\s*(\d+)",
        re.IGNORECASE
    )
    structure = []
    current_chapter = None
    current_section = None
    current_article = None

    virtual_chapter_created = False

    for page in pages_results:
        text = page["cleaned_text"]
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # ── Chapter Detection ─────────────────────
            ch_match = chapter_pattern.search(line)
            if ch_match:
                current_chapter = {
                    "chapter_number": ch_match.group(2),
                    "title": line,
                    "sections": []
                }
                structure.append(current_chapter)
                current_section = None
                current_article = None
                continue

            # ── Section Detection ─────────────────────
            sec_match = section_pattern.search(line)
            if sec_match:
                if not current_chapter:
                    # Create virtual chapter if missing
                    if not virtual_chapter_created:
                        current_chapter = {
                            "chapter_number": "0",
                            "title": "Virtual Chapter",
                            "sections": []
                        }
                        structure.append(current_chapter)
                        virtual_chapter_created = True

                current_section = {
                    "section_number": sec_match.group(2),
                    "title": line,
                    "articles": []
                }
                current_chapter["sections"].append(current_section)
                current_article = None
                continue

            # ── Article Detection ─────────────────────
            art_match = article_pattern.search(line)
            if art_match:
                if not current_chapter:
                    if not virtual_chapter_created:
                        current_chapter = {
                            "chapter_number": "0",
                            "title": "Virtual Chapter",
                            "sections": []
                        }
                        structure.append(current_chapter)
                        virtual_chapter_created = True

                if not current_section:
                    current_section = {
                        "section_number": "0",
                        "title": "General",
                        "articles": []
                    }
                    current_chapter["sections"].append(current_section)

                current_article = {
                    "article_number": art_match.group(2),
                    "title": line,
                    "clauses": []
                }
                current_section["articles"].append(current_article)
                continue

            # ── Clause / Text ─────────────────────────
            if current_article:
                current_article["clauses"].append(line)

    # If still nothing detected → wrap everything in virtual structure
    if not structure:
        structure = [{
            "chapter_number": "0",
            "title": "Virtual Chapter",
            "sections": [{
                "section_number": "0",
                "title": "General",
                "articles": [{
                    "article_number": "0",
                    "title": "Full Text",
                    "clauses": [
                        p["cleaned_text"] for p in pages_results if p["cleaned_text"]
                    ]
                }]
            }]
        }]

    return structure

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — JSON OUTPUT ASSEMBLY
#  Combine all per-page results into a single, well-structured JSON document.
# ═══════════════════════════════════════════════════════════════════════════════

def build_output(
    file_metadata: dict,
    pages_results: list[dict],
    processing_time_seconds: float,
) -> dict:
    """
    Assemble the final output JSON structure.

    Top-level keys:
        metadata        – file info + pipeline run info
        summary         – aggregate statistics across all pages
        pages           – per-page OCR results
    """
    total_words      = sum(p["word_count"]      for p in pages_results)
    total_characters = sum(p["character_count"] for p in pages_results)
    avg_confidence   = (
        round(sum(p["page_confidence"] for p in pages_results) / len(pages_results), 1)
        if pages_results else 0.0
    )
    has_arabic  = any(p["contains_arabic"]  for p in pages_results)
    has_english = any(p["contains_english"] for p in pages_results)

    if has_arabic and has_english:
        detected_languages = ["Arabic", "English"]
    elif has_arabic:
        detected_languages = ["Arabic"]
    elif has_english:
        detected_languages = ["English"]
    else:
        detected_languages = ["Unknown"]

    legal_structure = build_legal_structure(pages_results)

    return {
        "metadata": {
            **file_metadata,
            "pipeline_version": "2.0.0",
            "processing_time_seconds": round(processing_time_seconds, 2),
            "ocr_engine": "Tesseract",
        },
        "summary": {
            "total_pages": len(pages_results),
            "total_words": total_words,
            "total_characters": total_characters,
            "average_confidence": avg_confidence,
            "detected_languages": detected_languages,
        },
        "pages": pages_results,

        # ✅ NEW STRUCTURED OUTPUT
        "legal_structure": legal_structure
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    file_path: str,
    output_path: Optional[str] = None,
    language_override: Optional[str] = None,
    save_preprocessed: bool = False,
) -> dict:
    """
    Execute the complete OCR pipeline end-to-end.

    Parameters
    ----------
    file_path           : Path to the input PDF or image.
    output_path         : Where to write the JSON result.  If None, the result
                          is written to <input_stem>_ocr_output.json.
    language_override   : Force a specific Tesseract language string, e.g.
                          'ara', 'eng', or 'ara+eng'.  If None, auto-detected.
    save_preprocessed   : If True, save the pre-processed image(s) alongside
                          the JSON output for debugging.

    Returns
    -------
    dict : The complete structured OCR output.
    """
    start_time = time.time()

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    logger.info("━━━ Step 1/5 — Loading input file")
    pages, file_metadata = load_input(file_path)

    pages_results = []
    for page_index, pil_page in enumerate(pages, start=1):
        logger.info("━━━ Processing page %d / %d", page_index, len(pages))

        # ── Step 2: Pre-process ───────────────────────────────────────────────
        logger.info("  Step 2/5 — Pre-processing image")
        processed = preprocess_image(pil_page)

        if save_preprocessed:
            debug_path = f"preprocessed_page_{page_index}.png"
            cv2.imwrite(debug_path, processed)
            logger.info("  Saved pre-processed image → %s", debug_path)

        # ── Step 3: Language detection ────────────────────────────────────────
        if language_override:
            lang = language_override
            logger.info("  Step 3/5 — Language override: %s", lang)
        else:
            logger.info("  Step 3/5 — Detecting language")
            lang = detect_language(processed)
            logger.info("  Detected language: %s", lang)

        # ── Step 4: OCR ───────────────────────────────────────────────────────
        logger.info("  Step 4/5 — Running OCR (lang=%s)", lang)
        ocr_result = run_ocr(processed, lang=lang)

        # ── Step 5: Post-process ──────────────────────────────────────────────
        logger.info("  Step 5/5 — Post-processing text")
        ocr_result = postprocess_text(ocr_result)

        ocr_result["page_number"] = page_index
        pages_results.append(ocr_result)

        logger.info(
            "  ✔ Page %d done — %d words, confidence=%.1f%%",
            page_index, ocr_result["word_count"], ocr_result["page_confidence"],
        )

    # ── Assemble output ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    output = build_output(file_metadata, pages_results, elapsed)

    # ── Write JSON ────────────────────────────────────────────────────────────
    if output_path is None:
        stem = Path(file_path).stem
        output_path = f"{stem}_ocr_output.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("━━━ Pipeline complete in %.2fs → %s", elapsed, output_path)
    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OCR Pipeline — extract Arabic/English text from PDF or image files",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input",  help="Path to input PDF or image file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path for the JSON output file (default: <input>_ocr_output.json)",
    )
    parser.add_argument(
        "-l", "--lang",
        default=None,
        choices=["ara", "eng", "ara+eng"],
        help=(
            "Force language model:\n"
            "  ara     = Arabic only\n"
            "  eng     = English only\n"
            "  ara+eng = Both (default: auto-detect)"
        ),
    )
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save pre-processed page images for debugging",
    )
    parser.add_argument(
        "--print-text",
        action="store_true",
        help="Print extracted text to stdout after processing",
    )

    args = parser.parse_args()

    result = run_pipeline(
        file_path=args.input,
        output_path=args.output,
        language_override=args.lang,
        save_preprocessed=args.save_preprocessed,
    )

    if args.print_text:
        print("\n" + "═" * 60)
        print("EXTRACTED TEXT")
        print("═" * 60)
        for page in result["pages"]:
            print(f"\n--- Page {page['page_number']} ---")
            print(page["cleaned_text"])

    print("\nSummary:")
    s = result["summary"]
    print(f"  Pages         : {s['total_pages']}")
    print(f"  Words         : {s['total_words']}")
    print(f"  Characters    : {s['total_characters']}")
    print(f"  Avg confidence: {s['average_confidence']}%")
    print(f"  Languages     : {', '.join(s['detected_languages'])}")


if __name__ == "__main__":
    main()