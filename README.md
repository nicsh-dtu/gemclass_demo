# 📘 Publication Classification using the Google Gemini API

This repository contains a Jupyter notebook (`demo.ipynb`) that demonstrates how to classify academic publication PDFs using the **Google Gemini API**. The pipeline processes PDFs, applies classification using Gemini, and maps the output to predefined classes with optional error correction.

---

## 🌐 Setup: Google Gemini API Key

To use the Gemini API, you'll need to obtain an API key. Here's how:

### 🔑 Get Your API Key

1. Go to [Google Cloud Console](https://cloud.google.com/)
2. Click **Console** (top-right corner)
3. Click **Select a Project** → **New Project**
4. Visit [Google AI Studio](https://aistudio.google.com)
5. Click **Get API Key**
6. Save your API key securely (you’ll insert it into the notebook)

---

## 🧰 Files Overview

```
├── demo.ipynb                    # Main Jupyter notebook for classification
├── Setup/config.yaml             # Insert API-Key here
├── Setup/descriptions.yaml       # Maps class names to human-readable descriptions
├── Setup/strings_to_check.yaml   # List of strings that should invalidate a classification
├── Setup/wrong_classes.yaml      # Maps incorrect/hallucinated class names to correct ones
├── Files/                        # TODO: For classification, please create a folder 'Files', with respective subfolders (e.g. Project 1, Project 2), where you store the PDF's that you want to classify
├── requirements.txt              # Python dependencies
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/nicsh-dtu/gemclass_demo.git
cd gemclass_demo
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate         # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt     # Or run first notebook cell
```

### 4. Add Your API Key

Add your Gemini API-Key:

```yaml
api_key: "your-google-gemini-api-key"
```

> ⚠️ Never share/publish your key.


### 5. Add the folder 'Files', that contains subfolders (e.g. 'Project 1') with relevant PDF's.

---

## 📄 Usage Instructions

1. Place your PDF files into the `Files/SUBFOLDERS` directory
2. Open `demo.ipynb` in JupyterLab or VS Code
3. Run through the cells:
   - Loads PDFs
   - Sends requests to Gemini API
   - Classifies the content based on `descriptions.yaml`
   - Applies filtering rules via `strings_to_check.yaml`
   - Corrects hallucinated class names via `wrong_classes.yaml`

---

## ⚙️ Customization

- **`descriptions.yaml`**  
  Define the possible classes and their descriptions. Gemini will use these descriptions to match publications.

- **`strings_to_check.yaml`**  
  Contains phrases that, if found in Gemini's output, cause the classification to be discarded.

- **`wrong_classes.yaml`**  
  Fixes hallucinated or incorrect class names by mapping them to valid ones.

---

## 📄 License

This code is provided for educational and demonstration purposes related to seminars hosted by the Technical University of Denmark. Attendees and colleagues are permitted to use, adapt, and reference this code for non-commercial, internal, and educational use only. Redistribution or commercial use is prohibited without explicit permission from the authors.
