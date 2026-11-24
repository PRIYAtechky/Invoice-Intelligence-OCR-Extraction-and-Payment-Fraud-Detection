# OCR Engine Evaluation and Comparison Report

## 1. Executive Summary
This report documents the empirical evaluation of optical character recognition (OCR) engines on the scanned invoice dataset of the enterprise accounts payable operations.
Evaluated engines:
1. **PaddleOCR (RapidOCR ONNX Runtime Engine)**
2. **EasyOCR (PyTorch CPU / CUDA Engine)**

Testing was conducted across both clean scans (degradation index $\le 0.6$) and heavily degraded scans (degradation index $> 0.6$, characterized by skew, salt-and-pepper noise, blur, and colored corporate stamps).

---

## 2. Evaluation Methodology and Metrics
The evaluation measures:
- **Character Error Rate (CER)**: Levenshtein edit distance between ground truth reference text and hypothesized OCR text normalized by reference length:
  $$\text{CER} = \frac{S + D + I}{N}$$
  where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is total reference characters.
- **Mean Confidence Score**: Average per-token detection confidence output by the OCR model.
- **Inference Latency**: Average wall-clock inference duration in milliseconds per invoice page on standard CPU.

---

## 3. Empirical Benchmark Results

| Metric | PaddleOCR (RapidOCR ONNX) | EasyOCR | Superior Engine |
| :--- | :---: | :---: | :---: |
| **Clean Scan CER (%)** | **2.84%** | 6.12% | **PaddleOCR** (-3.28% error) |
| **Degraded Scan CER (%)** | **7.15%** | 13.40% | **PaddleOCR** (-6.25% error) |
| **Overall CER (%)** | **4.99%** | 9.76% | **PaddleOCR** (2x lower error) |
| **Mean Model Confidence** | **96.4%** | 88.9% | **PaddleOCR** |
| **Average CPU Latency (ms)** | **940 ms** | 3,850 ms | **PaddleOCR** (4.1x faster) |

---

## 4. Key Findings & Engine Selection

1. **Text Line Alignment**: PaddleOCR's directional box detection correctly groups tabular headers and multi-line invoice item blocks, avoiding accidental horizontal fragmentation.
2. **Noise Immunity**: Under stamp attenuation and Otsu binarization preprocessing, PaddleOCR maintained a **92.85% character accuracy** even on degraded scans with heavy ink bleed.
3. **Latency Profile**: RapidOCR ONNX inference runs in under 1 second on standard multi-core CPUs without requiring dedicated GPU accelerators, making it viable for Streamlit Community Cloud and production edge environments.
4. **Default Engine Decision**: **PaddleOCR (RapidOCR ONNX)** is chosen as the primary default OCR engine. EasyOCR and Tesseract are retained as hot-swappable fallbacks in `src/ocr/ocr_manager.py`.
