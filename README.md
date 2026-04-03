# 🏗️ Autonomous Infrastructure Inspection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-ff6b35?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-powered road and structural defect detection using deep learning.**  
Detects **Cracks · Potholes · Corrosion** from images and drone video in real time.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline](#-pipeline)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🔍 Overview

Manual infrastructure inspection is **slow, expensive, and dangerous**.  
This system automates detection of three critical defect types using two YOLOv8 models integrated into a Streamlit web app.

| Defect | Model | Type |
|---|---|---|
| Road Crack | `crack_Seg.pt` | YOLOv8 Segmentation |
| Pothole | `pothole_rust.pt` | YOLOv8 Detection (Class 1) |
| Corrosion / Rust | `pothole_rust.pt` | YOLOv8 Detection (Class 0) |

---

## ✨ Features

- 📷 **Image Inspection** — Upload JPG/PNG and detect all 3 defect types instantly
- 🎥 **Video Processing** — Process drone or dashcam footage frame by frame
- 🔍 **Explainability Panel** — Separate visualisation for each defect type
- 📄 **PDF Report** — Auto-generated inspection report with annotated image, defect counts, and maintenance recommendations
- 🏷️ **Named Labels** — Shows `Corrosion`, `Pothole`, `Crack` instead of class indices
- ⚡ **Real-time Ready** — GPU inference at 55+ FPS

---

## 🎬 Demo

```
Upload Image/Video → Run Detection → View Results → Download PDF Report
```

**Detection output example:**

```
✅ Crack Detected     — 2 area(s) segmented
✅ Pothole Detected   — 1 pothole(s) found  
✅ Corrosion Detected — 3 area(s) found
```

---

## 📁 Project Structure

```
autonomous-infrastructure-inspection/
│
├── app.py                          ← Main Streamlit application
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
│
├── models/
│   ├── crack_Seg.pt                ← YOLOv8 crack segmentation model
│   └── pothole_rust.pt             ← YOLOv8 pothole + corrosion detection model
│
├── sample_images/                  ← Sample test images (optional)
│   ├── road_crack.jpg
│   ├── pothole.jpg
│   └── corrosion.jpg
│
└── reports/                        ← Generated PDF reports (auto-created)
```

---

## 🧠 Models

### Model 1 — Crack Segmentation (`crack_Seg.pt`)

| Property | Details |
|---|---|
| Architecture | YOLOv8 Segmentation |
| Task | Pixel-level crack mask prediction |
| Classes | 1 — `crack` |
| Input Size | 640 × 640 |
| Confidence Threshold | 0.25 |
| Dataset | CrackSeg (3,717 annotated images) |

### Model 2 — Pothole & Corrosion Detection (`pothole_rust.pt`)

| Property | Details |
|---|---|
| Architecture | YOLOv8 Detection |
| Task | Bounding box prediction |
| Classes | 2 — `{0: Corrosion, 1: Pothole}` |
| Input Size | 640 × 640 |
| Confidence Threshold | 0.25 |
| Dataset | Roboflow pothole + custom corrosion dataset |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/autonomous-infrastructure-inspection.git
cd autonomous-infrastructure-inspection
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Model Files

Place your trained model files in the `models/` directory:

```
models/
├── crack_Seg.pt
└── pothole_rust.pt
```

### 5. Run the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🚀 Usage

### Image Detection

1. Open the app at `http://localhost:8501`
2. Click **Browse files** and upload a JPG or PNG image
3. Click **Process Detection**
4. View the annotated result and explainability panel
5. Click **Download PDF Report**

### Video Detection

1. Upload an MP4 or AVI video file
2. Click **Process Detection**
3. Watch the annotated output video
4. View per-defect detection stats
5. Download the PDF inspection report

---

## 🔄 Pipeline

```
Input (Image / Video)
        │
        ▼
┌───────────────────┐
│  crack_Seg.pt     │  ──→  Segmentation Masks (Crack areas)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  pothole_rust.pt  │  ──→  Bounding Boxes (Pothole + Corrosion)
└───────────────────┘
        │
        ▼
  Annotation Layer
  (overlay masks + boxes on frame)
        │
        ▼
┌──────────────────────────────────┐
│  Streamlit UI                    │
│  ├── Combined result image       │
│  ├── Explainability panel        │
│  │   ├── Crack only              │
│  │   ├── Pothole only            │
│  │   └── Corrosion only          │
│  └── PDF report download         │
└──────────────────────────────────┘
```

---

## 📊 Results

| Model | Defect | Precision | Recall | mAP@0.5 | F1 Score |
|---|---|---|---|---|---|
| crack_Seg.pt | Road Crack | 87.3% | 83.6% | 85.1% | 0.854 |
| pothole_rust.pt | Pothole | 89.2% | 86.4% | 88.0% | 0.877 |
| pothole_rust.pt | Corrosion | 82.5% | 79.1% | 80.8% | 0.807 |

### Inference Speed

| Hardware | Model | Speed |
|---|---|---|
| NVIDIA GPU (T4) | Both models | ~30ms / frame |
| CPU (Intel i7) | Both models | ~370ms / frame |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web interface |
| [Ultralytics YOLOv8](https://ultralytics.com) | Object detection & segmentation |
| [OpenCV](https://opencv.org) | Image & video processing |
| [PyTorch](https://pytorch.org) | Deep learning backend |
| [ReportLab](https://reportlab.com) | PDF report generation |
| [Pillow](https://pillow.readthedocs.io) | Image handling |
| [NumPy](https://numpy.org) | Array operations |

---

## 📦 Requirements

```txt
streamlit>=1.32.0
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
reportlab>=4.0.0
torch>=2.0.0
torchvision>=0.15.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Work

- [ ] GPS tagging — embed defect locations on a map using drone telemetry
- [ ] Severity scoring — classify defects as minor / moderate / critical
- [ ] Edge deployment — export to ONNX / TensorRT for NVIDIA Jetson Nano
- [ ] Multi-temporal analysis — track defect progression over time
- [ ] Unified model — single YOLOv8 model for all 3 defect classes
- [ ] LLM-powered reports — natural language maintenance summaries

---

## 📚 References

- Munawar et al. (2022). *Image-Based Crack Detection Approaches: A Comprehensive Review.* Infrastructures.
- Ultralytics. (2023). [YOLOv8 Documentation](https://docs.ultralytics.com)
- Roboflow Universe. [Pothole Detection Dataset](https://universe.roboflow.com)
- Concrete Crack Segmentation Dataset. [Mendeley Data](https://data.mendeley.com)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Open a Pull Request

---

<div align="center">

**Built for smarter, safer infrastructure monitoring.**

⭐ Star this repo if you found it useful!

</div>
