# ⚽ SoccerVision — Player Tracking & Re-Identification

> A hybrid computer-vision pipeline for detecting, tracking, and re-identifying soccer players across video frames using **YOLO, Deep SORT, OSNet, appearance embeddings, jersey-color analysis, and motion cues**.

---

## 🎯 Overview

Tracking players in a soccer match is more challenging than simply detecting people in individual frames.

Players constantly:

* enter and leave the camera view
* overlap during tackles and set pieces
* become temporarily occluded
* change position rapidly
* appear visually similar because teammates wear identical kits
* lose their tracking identity when a tracker can no longer associate them with previous observations

**SoccerVision** addresses these challenges with a hybrid detection, tracking, and re-identification pipeline.

Instead of relying exclusively on a conventional multi-object tracker, the system combines several independent signals to recover player identities when tracking continuity is broken.

### Core pipeline

```text
                Input Video
                     │
                     ▼
          ┌─────────────────────┐
          │   YOLO Detection    │
          │ Players + Ball      │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │     Deep SORT       │
          │ Short-term Tracking │
          └──────────┬──────────┘
                     │
              Track Lost?
                /        \
              No          Yes
              │            │
              │            ▼
              │    ┌───────────────┐
              │    │ Hybrid Re-ID  │
              │    └───────┬───────┘
              │            │
              │     ┌──────┼──────┐
              │     ▼      ▼      ▼
              │  OSNet   Motion  Color
              │ Features Similarity Similarity
              │     │      │      │
              │     └──────┼──────┘
              │            ▼
              │      Match Scoring
              │            │
              └────────────┤
                           ▼
                  Consistent Player IDs
                           │
                           ▼
                    Annotated Video
```

---

# ✨ Key Features

### 🧠 Deep Player Re-Identification

Uses **OSNet-based person re-identification embeddings** to compare the appearance of players across frames.

Historical embeddings are maintained for tracks so that a player can potentially be recovered after temporary tracking failure.

### 🔄 Deep SORT Tracking

Deep SORT provides short-term multi-object tracking and maintains player identities while visual and spatial continuity is available.

### 👕 Jersey Appearance Analysis

Dominant jersey color is extracted from player regions using color clustering.

This provides an additional appearance cue when multiple players have similar body-level features.

### 🧭 Motion-Aware Matching

Spatial information is incorporated into re-identification.

The system considers:

* recent player trajectories
* predicted positions
* previous player locations
* distance between predicted and observed positions

This helps reject visually plausible but spatially unlikely matches.

### ⚽ Player & Ball Detection

The YOLO model is used as the detection stage for the soccer footage, providing object detections that feed the tracking pipeline.

### 🧩 Hybrid Matching

Instead of depending on a single metric, multiple signals are combined:

```text
Appearance
    +
Motion
    +
Color
    ↓
Re-Identification Score
```

This makes the system more robust to temporary occlusion and track fragmentation.

---

# 🏗️ Project Structure

```text
Soccer-Reidentification/
│
├── model2.py
├── reid_functions.py
├── requirements.txt
│
├── deep_sort_pytorch/
│   └── ...
│
├── soccer_yolov11.pt
├── 15sec_input_720p.mp4
├── output_video.mp4
│
├── .gitignore
└── README.md
```

> Large model weights and video files are intentionally excluded from the Git repository where appropriate.

---

# 🔬 How It Works

## 1. Object Detection

Each video frame is passed through the YOLO detection model.

The detector identifies relevant objects such as:

```text
Player
Ball
```

The resulting bounding boxes are passed into the tracking stage.

---

## 2. Short-Term Tracking

Detected players are processed by **Deep SORT**.

Deep SORT associates detections across consecutive frames using spatial and appearance information.

When a player remains continuously visible, the tracker can maintain the same ID without requiring expensive re-identification on every frame.

Example:

```text
Frame 100   → Player 7
Frame 101   → Player 7
Frame 102   → Player 7
Frame 103   → Player 7
```

---

## 3. Track Fragmentation

A tracking identity can be lost when a player:

```text
             ┌─────────────┐
             │ Player View │
             └──────┬──────┘
                    │
             temporary occlusion
                    │
                    ▼
             ┌─────────────┐
             │ Track Lost  │
             └──────┬──────┘
                    │
                    ▼
             Player Reappears
                    │
                    ▼
              Hybrid Re-ID
                    │
                    ▼
             Previous ID?
```

Instead of immediately treating the returning player as a completely new identity, the system evaluates previously lost tracks.

---

# 🧬 Re-Identification

The re-identification stage combines three complementary signals.

## 1. Appearance Similarity

OSNet generates a feature representation for each detected player.

Conceptually:

```text
Player Crop
     │
     ▼
   OSNet
     │
     ▼
Feature Embedding
     │
     ▼
Similarity Comparison
```

Historical feature observations are retained for tracks.

When a new detection appears, its embedding can be compared with the stored representation of previously lost players using cosine similarity.

---

## 2. Jersey Color Similarity

Players from the same team can have very similar visual appearances.

A dominant-color representation provides another lightweight visual cue.

The pipeline extracts representative RGB color information from the player crop and compares it against the historical appearance of the track.

Conceptually:

```text
Player Crop
    │
    ▼
Color Clustering
    │
    ▼
Dominant Color
    │
    ▼
Distance / Similarity
```

Color is treated as a supporting signal rather than the sole identity criterion.

---

## 3. Motion Similarity

Visual similarity alone can produce incorrect matches.

For example:

```text
Player A → visually similar
Player B → visually similar

Current detection is physically close to Player A
but far from Player B
```

Motion information can therefore help resolve ambiguity.

Depending on available tracking information, the system can use predicted or historical positions to estimate how plausible a candidate match is.

---

# 🧮 Matching Strategy

The final matching decision combines the available signals.

The implemented scoring formulation is:

```text
Score =
    0.5 × Appearance Similarity
  + 0.4 × Motion Similarity
  + 0.2 × Color Similarity
```

The individual components contribute different information:

| Signal         | Purpose                 |
| -------------- | ----------------------- |
| Appearance     | Visual identity         |
| Motion         | Spatial consistency     |
| Color          | Jersey-level appearance |
| Combined Score | Candidate ranking       |

> The weights are implementation parameters and can be tuned for different footage and camera configurations.

---

# 🛠️ Technology Stack

| Technology       | Role                                   |
| ---------------- | -------------------------------------- |
| **Python**       | Core implementation                    |
| **YOLOv11**      | Object detection                       |
| **Deep SORT**    | Multi-object tracking                  |
| **OSNet**        | Person re-identification               |
| **TorchReID**    | Re-ID model interface                  |
| **OpenCV**       | Video processing                       |
| **NumPy**        | Numerical operations                   |
| **scikit-learn** | Color clustering / supporting analysis |

---

# 🚀 Getting Started

## Requirements

Recommended environment:

```text
Python 3.x
PyTorch
CUDA-enabled GPU (recommended)
```

A GPU is strongly recommended for practical inference speed, particularly when extracting deep Re-ID features.

---

## 1. Clone

```bash
git clone https://github.com/naresh04122005/Soccer-Reidentification.git
cd Soccer-Reidentification
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Install TorchReID if it is not already included in your environment:

```bash
pip install torchreid
```

---

# 📦 Model Weights

The YOLO model weights are larger than GitHub's standard individual-file limit and therefore should not be committed directly to the repository.

### Download

**YOLO Soccer Detection Model**

[Download `soccer_yolov11.pt`](https://drive.google.com/uc?export=download&id=1kQCbXQqg3C9DXtllPS5WgMVOl3kaRkYH)

After downloading, place the model in the project root:

```text
Soccer-Reidentification/
└── soccer_yolov11.pt
```

Make sure the filename matches the path expected by the inference code.

---

# 🎥 Input Video

The example pipeline uses:

```text
15sec_input_720p.mp4
```

Place the input video in the project directory if it is not already available.

For larger datasets or videos, consider storing them outside Git or using Git LFS / external object storage.

---

# ▶️ Running the Pipeline

Run:

```bash
python model2.py
```

The pipeline processes the configured input video and generates the corresponding annotated output.

Example:

```text
Input
└── 15sec_input_720p.mp4

        │
        ▼

Detection
        │
        ▼

Deep SORT
        │
        ▼

Hybrid Re-ID
        │
        ▼

Output
└── output_video.mp4
```

---

# 📊 Output

The generated video contains:

* player detections
* tracking bounding boxes
* player IDs
* re-identified tracks

The primary objective is to maintain a more consistent identity assignment when players temporarily disappear or become occluded.

---

# ⚠️ Current Limitations

This system is designed as a research/experimental pipeline and is not intended to solve every soccer-tracking scenario.

Performance can degrade under:

* severe player occlusion
* extremely small player crops
* heavy motion blur
* abrupt camera movement
* large viewpoint changes
* visually identical players
* long periods where a player remains completely outside the frame
* crowded penalty-box scenes

Jersey color can also become unreliable under changing illumination, shadows, compression artifacts, or similar team kits.

---

# 🔮 Roadmap

Potential improvements include:

### 🧤 Goalkeeper Identification

Use spatial position, jersey appearance, and detection characteristics to distinguish goalkeepers from outfield players.

### 🔢 Jersey Number Recognition

Integrate OCR to extract jersey numbers and use them as a high-confidence identity cue.

Possible pipeline:

```text
Player Detection
      │
      ▼
Jersey Region
      │
      ▼
OCR
      │
      ▼
Jersey Number
      │
      ▼
Identity Verification
```

### ⚽ Ball & Player Interaction

Extend the pipeline to detect events such as:

* player possession
* passes
* ball recovery
* shots
* player-ball proximity

### 📐 Improved Re-ID

Explore stronger metric-learning and temporal aggregation techniques instead of relying primarily on manually weighted similarity components.

### ⚡ Real-Time Inference

Investigate:

* ONNX
* TensorRT
* GPU batching
* model quantization
* asynchronous video processing

### 🖥️ Interactive Interface

Add a lightweight Streamlit or Gradio interface allowing users to:

1. upload a match video
2. select models
3. configure thresholds
4. run inference
5. preview the resulting tracking output

### ⚙️ Configuration System

Move hard-coded parameters into a configuration file:

```text
config/
└── default.yaml
```

This would make experiments easier to reproduce and compare.

---

# 🧪 Reproducibility

For consistent experiments, keep track of:

```text
Detection Model
Re-ID Model
Confidence Threshold
Tracking Parameters
Re-ID Weights
Input Resolution
GPU / CPU Configuration
```

When comparing experiments, change one major parameter at a time whenever possible.

---

# 📝 Development Notes

The project intentionally separates the main inference pipeline from supporting Re-ID utilities.

### `model2.py`

Main orchestration layer responsible for:

* video processing
* detection
* tracking
* identity management
* output generation

### `reid_functions.py`

Supporting functionality for:

* feature extraction
* appearance comparison
* color analysis
* IoU calculations
* motion similarity
* Re-ID utilities

### `deep_sort_pytorch/`

Deep SORT implementation used by the tracking pipeline.

---

# 📁 Large Files

Avoid committing large generated or binary assets directly to Git.

Examples:

```text
*.pt
*.pth
*.mp4
*.avi
*.mov
```

For large assets, use:

* Git LFS
* Google Drive
* Hugging Face Hub
* cloud object storage

and document the download location in this README.

---

# 🔐 Recommended `.gitignore`

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
env/

*.pt
*.pth

*.mp4
*.avi
*.mov
*.mkv

output/
outputs/

.ipynb_checkpoints/

.vscode/
.idea/

.DS_Store
```

---

# 📜 License

This project is intended for **academic and research purposes**.

Before redistributing model weights, datasets, or third-party components, verify the applicable licenses and usage restrictions.

---

# 👨‍💻 Author

**Naresh Sihag**

GitHub: `@naresh04122005`

---

## ⭐ Project Summary

**SoccerVision** combines object detection, multi-object tracking, and appearance-based re-identification into a unified pipeline for soccer video analysis.

The central idea is simple:

```text
Detection alone
       ↓
Tracking alone
       ↓
       ❌

Detection
    +
Tracking
    +
Appearance
    +
Jersey Color
    +
Motion
    ↓
More Robust Player Re-Identification
```

The project provides a foundation for building more advanced soccer analytics systems involving player identity, trajectories, team analysis, and player-ball interactions.
