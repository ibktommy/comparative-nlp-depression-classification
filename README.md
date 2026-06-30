# Multi-Class Depression Severity Classification & Comparative Inference Dashboard

### An interactive application comparing classical machine learning, deep learning, and transformer models for text analysis.

**Live Application Link:** [https://multi-class-depression-predictive-analysis.streamlit.app/](https://multi-class-depression-predictive-analysis.streamlit.app/)

---

## 📌 Project Overview
In this project, I built an end-to-end classification system that analyzes text to determine the severity level of a person's depression. I developed a complete pipeline that cleans raw text data, extracts linguistic features, and processes sentences so they can be evaluated by three distinct artificial intelligence frameworks.

My final testing shows that all three models perform highly when classifying different severity levels. By comparing a straightforward statistical model, a pattern-tracking deep learning model, and an advanced contextual transformer, I successfully deployed a parallel dashboard that lets users test and compare all models simultaneously in real time.

---

## 🛠️ System Architecture & Workflow

```text
📊 Raw Text Input
       │
       ├───> [TF-IDF Text Feature Matrix] ───────> Hybrid LR / SVM Pipeline
       │
       ├───> [Text Tokenizer & Max-Padding] ─────> Bidirectional GRU Network
       │
       └───> [DistilBERT Sub-word Tokenizer] ────> Fine-Tuned DistilBERT Transformer
```

---

## 🔬 Implementation Details

### 1. Data Preprocessing & Feature Engineering
* **TF-IDF Configuration:** I converted raw words into numerical scores based on their frequency and uniqueness across the dataset, helping the traditional machine learning models identify mathematical boundaries between categories.
* **GRU Token Embedding:** I transformed text sentences into organized lists of numbers using a custom word dictionary, ensuring every sentence was cropped or padded to the exact same length so the neural network could process them as uniform matrices.
* **Transformer Tokenization:** I utilized HuggingFace's specialized `DistilBertTokenizerFast` tool to break complex words down into smaller sub-word roots, ensuring the input matches exactly what the pre-trained transformer model expects.

### 2. Model Deep Dives

#### Hybrid LR & SVM Components
* I optimized multi-class Logistic Regression and Support Vector Machine models to serve as highly efficient, mathematically straightforward baselines that prioritize fast execution speeds.
* I saved these components as permanent pipeline files (`tfidf_vectorizer.pkl`, `hybrid_lr_component.pkl`, `hybrid_svm_model.pkl`) to make sure the exact same text-processing steps are mirrored on the server.

#### Optimized Gated Recurrent Unit (GRU)
* I built a Bidirectional GRU network, which is a specialized deep learning architecture designed to read sentences forward and backward to capture the full context of a statement without losing track of long-range details.
* I added dropout layers to prevent overfitting and integrated a global pooling layer to pass the most prominent text features directly into a final multi-neuron classification layer.
* I compiled and saved this trained deep learning graph as a local file named `gru_optimized_model.h5`.

#### Fine-Tuned DistilBERT
* I adapted the industry-standard `distilbert-base-uncased` transformer architecture, which uses self-attention mechanisms to understand the deeper, underlying meaning of sentences while remaining lightweight and fast.
* I fine-tuned the model by adjusting its weights on my specific mental health text dataset, training it to map complex emotional language directly to the correct depression severity index.
* I saved the final fine-tuned model weights inside a dedicated repository folder structure named `distilbert_saved_model/`.

---

## 🚀 Repository Directory Structure

> 💡 **Important Note on File Locations:** To keep this GitHub repository lightweight and comply with storage limits, **I do not store the heavy model weights directly on GitHub**. Instead, I have split the project assets strategically across three environments: **GitHub** hosts the core codebase, **Hugging Face Hub** permanently stores the heavy trained model files, and **Streamlit Cloud** acts as the runtime server that pulls everything together automatically on boot. 

The structure below illustrates what the virtual environment looks like on the Streamlit server *after* it pulls my heavy assets from Hugging Face:

```text
├── downloaded_models/               # 💾 HOUSES TRAINED WEIGHTS (Mirrored from Hugging Face on boot)
│   ├── tfidf_vectorizer.pkl         # Saved text-to-number converter
│   ├── hybrid_lr_component.pkl      # Trained Logistic Regression model
│   ├── hybrid_svm_model.pkl         # Trained SVM classification model
│   ├── gru_optimized_model.h5       # Saved Deep Learning GRU network graph
│   ├── tokenizer_gru.pkl            # Word-to-number dictionary for the GRU
│   ├── label_encoder_gru.pkl        # Text category converter matrix
│   └── distilbert_saved_model/      # Fine-Tuned Transformer Weights
│       ├── config.json              # Structural transformer blueprint file
│       └── model.safetensors        # Fine-tuned classification parameters file
├── app.py                           # 📂 ON GITHUB: Main Streamlit parallel inference script
├── app_test_samples.csv             # 📂 ON GITHUB: 100 random text samples for instant testing
├── requirements.txt                 # 📂 ON GITHUB: List of python packages required to run this app
└── README.md                        # 📂 ON GITHUB: This documentation file
```

---

## ⚡ Setup, Installation & Dashboard Execution

Because large model weights can cause deployment timeouts, I designed the interactive web dashboard to automatically download the heavy deep learning and transformer weights directly from a HuggingFace hub repository snapshot during its initialization phase.

### Local Setup

1. **Clone the Repository:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

2. **Configure Virtual Environment & Install Required Modules:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Launch the Dashboard Application Locally:**
```bash
streamlit run app.py
```

### Production Deployment Strategy
When deploying this system directly to cloud environments like **Streamlit Cloud**, I wrote the internal codebase to automatically verify the server environment. If the local text-processing files are missing from the folder directories, the script automatically redirects its download commands to pull the standard base text components directly from the official HuggingFace public registry, preventing application crashes and ensuring smooth deployment uptime.