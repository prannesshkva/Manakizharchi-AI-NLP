# Manakizharchi-AI-NLP

<p align="center">
  <strong>Multimodal Tamil Sentiment & Affective Cognitive Intelligence Engine</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge&logo=github&logoColor=white" alt="Status" />
  <img src="https://img.shields.io/badge/Language-Python_3.10+-18181B?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Domain-Tamil_NLP_&_Emotion-18181B?style=for-the-badge" alt="Domain" />
  <img src="https://img.shields.io/badge/License-MIT-18181B?style=for-the-badge" alt="License" />
</p>

---

## Overview

**Manakizharchi-AI-NLP** (*மனக்கிளர்ச்சி* — Affective Excitation / Emotional State) is an end-to-end natural language and cognitive computing pipeline designed to process, classify, and reason over vernacular Tamil text, psychological documents, and emotional sentiment corpora.

The system integrates document digitization via optical character recognition (OCR), dense semantic embeddings, vector indexing with ChromaDB, hybrid TF-IDF retrieval, and multi-model classification ensembles to enable deep affective sentiment reasoning.

---

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-18181B?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Hugging_Face-18181B?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Sentence_Transformers-18181B?style=for-the-badge&logo=pytorch&logoColor=white" alt="Sentence Transformers" />
  <img src="https://img.shields.io/badge/ChromaDB-18181B?style=for-the-badge&logo=databricks&logoColor=white" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Scikit_Learn-18181B?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Tesseract_OCR-18181B?style=for-the-badge&logo=google&logoColor=white" alt="OCR" />
</p>

---

## Architectural Pipeline

```
Raw Tamil Documents / PDFs / Text Inputs
                   │
                   ▼
       [Optical Character Recognition]  (pytesseract, pdfplumber, pdf2image)
                   │
                   ▼
         [Text Preprocessing]           (dataprocessing.py - Tokenization & Cleansing)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
[Dense Embeddings]      [Sparse Lexical Scoring]
(SentenceTransformers)      (TF-IDF Vectorizer)
        │                     │
        ▼                     ▼
[ChromaDB Vector Store] [Hybrid Retrieval Engine]
        └──────────┬──────────┘
                   │
                   ▼
      [Multi-Model Ensemble]            (multimodel_training.py)
                   │
                   ▼
      [LLM Affective Reasoning]        (llmanswering.py)
```

---

## Key Capabilities

- **Vernacular & Multilingual Processing**: Engineered to navigate Tamil syntactic nuances, sentiment shifts, and colloquial formulations.
- **Multimodal Document Parsing**: Converts unstructured PDFs and scanned documents into structured lexical corpora via Tesseract OCR and PDFPlumber.
- **Hybrid Retrieval Architecture**: Combines exact lexical scoring (TF-IDF) with dense contextual semantic embeddings to prevent semantic drift.
- **Affective State Classification**: Classifies affective arousal and emotional polarity across diverse textual contexts.

---

## Repository Structure

```
Manakizharchi-AI-NLP/
├── .devcontainer/             # Visual Studio Code development container config
│   └── devcontainer.json
├── dataprocessing.py          # Document extraction, OCR, and text cleansing
├── dependencies.py            # Environment validation and library bootstrapping
├── llmanswering.py            # Generative LLM answering and context synthesis
├── main.py                    # Pipeline execution and orchestration entrypoint
├── multimodel_training.py     # Training and evaluation of multi-model classifiers
└── README.md
```

---

## Quickstart

### 1. System Dependencies (Tesseract OCR)
* **Ubuntu/Debian**:
  ```bash
  sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-tam poppler-utils
  ```
* **macOS**:
  ```bash
  brew install tesseract tesseract-lang poppler
  ```

### 2. Python Environment Setup
```bash
git clone https://github.com/prannesshkva/Manakizharchi-AI-NLP.git
cd Manakizharchi-AI-NLP
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install sentence-transformers chromadb scikit-learn pdfplumber pytesseract pdf2image nltk requests pandas numpy
```

### 3. Pipeline Execution
```bash
python main.py
```

---

## Author

* **Author**: [Prannessh KVA](https://github.com/prannesshkva)
* **Focus**: Multilingual NLP, Vernacular Affective Computing, and Retrieval Architectures

---

## License

This project is licensed under the MIT License.
