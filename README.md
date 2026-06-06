# Arabic & English OCR Pipeline for Legal Document Processing

## Overview

A production-ready OCR pipeline designed for extracting and structuring text from Arabic and English legal documents.

The system supports both scanned PDFs and image files, performs advanced image preprocessing, automatically detects document language, extracts text using Tesseract OCR, and converts unstructured legal documents into a hierarchical JSON format suitable for NLP, Information Retrieval, RAG systems, and Legal AI applications.

The pipeline is specifically optimized for low-quality scans, bilingual documents, legal contracts, regulations, corporate laws, and government documents.

---

## Key Features

### Multi-Format Support

* PDF documents
* PNG
* JPG / JPEG
* TIFF
* BMP
* WebP

### Advanced Image Preprocessing

* Image upscaling
* CLAHE contrast enhancement
* Watermark suppression
* Noise removal
* Otsu binarization
* Morphological repair of broken text
* Automatic deskewing

### Automatic Language Detection

Detects:

* Arabic
* English
* Mixed Arabic-English documents

Automatically selects the appropriate OCR model.

### OCR Extraction

Extracts:

* Full page text
* Word-level text
* Bounding boxes
* Confidence scores
* Block structure
* Paragraph structure
* Line structure

### Legal Document Structuring

Converts raw OCR output into:

Chapter → Section → Article → Clause

Supports:

* Arabic legal documents
* English legal documents
* Mixed bilingual legal documents

### Rich JSON Output

Provides:

* OCR text
* Layout information
* Confidence scores
* Language metadata
* Legal hierarchy extraction

---

## Pipeline Architecture

Input File
↓
Document Loading
↓
Image Preprocessing
↓
Language Detection
↓
OCR Extraction
↓
Text Postprocessing
↓
Legal Structure Parsing
↓
JSON Output

---

## Technologies Used

### OCR

* Tesseract OCR

### Computer Vision

* OpenCV
* NumPy

### PDF Processing

* pdf2image
* Poppler

### Language Detection

* langdetect

### Image Processing

* Pillow

### Programming Language

* Python 3.10+

---

## Installation

### Install Python Dependencies

```bash
pip install pytesseract pdf2image opencv-python-headless pillow numpy langdetect
```

### Install Tesseract

#### macOS

```bash
brew install tesseract
brew install tesseract-lang
```

#### Ubuntu

```bash
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-ara
```

### Install Poppler

#### macOS

```bash
brew install poppler
```

#### Ubuntu

```bash
sudo apt install poppler-utils
```

---

## Usage

### Basic Usage

```bash
python ocr_pipeline.py document.pdf
```

### Specify Output File

```bash
python ocr_pipeline.py document.pdf -o output.json
```

### Force Language

```bash
python ocr_pipeline.py document.pdf --lang ara
```

```bash
python ocr_pipeline.py document.pdf --lang eng
```

```bash
python ocr_pipeline.py document.pdf --lang ara+eng
```

### Save Preprocessed Images

```bash
python ocr_pipeline.py document.pdf --save-preprocessed
```

### Print Extracted Text

```bash
python ocr_pipeline.py document.pdf --print-text
```

---

## Output Structure

```json
{
  "metadata": {},
  "summary": {},
  "pages": [],
  "legal_structure": []
}
```

### Metadata

Contains:

* File information
* Processing time
* OCR engine details
* Pipeline version

### Summary

Contains:

* Total pages
* Total words
* Total characters
* Average OCR confidence
* Detected languages

### Pages

Contains:

* Full text
* Cleaned text
* Word-level OCR results
* Bounding boxes
* Confidence scores

### Legal Structure

Contains:

```text
Chapter
 └── Section
      └── Article
            └── Clause
```

---

## Use Cases

* Legal AI systems
* Retrieval-Augmented Generation (RAG)
* Corporate Law Chatbots
* Regulatory Compliance Systems
* Knowledge Base Construction
* Document Digitization
* Government Record Processing
* Contract Analysis

---

## Future Improvements

* PaddleOCR integration
* Docling integration
* Table extraction
* LayoutLM support
* Named Entity Recognition (NER)
* Semantic legal clause classification
* Arabic legal language normalization
* Vector database integration for RAG systems

---

## License

MIT License
