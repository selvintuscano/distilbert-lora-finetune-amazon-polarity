# 🚀 Finetune DistilBERT with LoRA on Amazon Polarity Dataset
**Efficient Sentiment Analysis using PEFT, LoRA, and Streamlit Deployment**

---
Project Resources
GitHub Repository: https://github.com/selvintuscano/distilbert-lora-finetune-amazon-polarity
Presentations:
1.	LoRA and PEFT Explanation: https://drive.google.com/file/d/1G0e2izyMiGLTqED9_qmPLS0LnjILX698/view
2.	Code and Project Walkthrough: https://drive.google.com/file/d/1GHThc9iHC9uYcRtAeAyXo9LrPFqAThQc/view

---
## 📖 Overview
This project demonstrates how to fine-tune **DistilBERT** using **LoRA (Low-Rank Adaptation)** and Hugging Face's **PEFT** library on the **Amazon Polarity dataset** for binary sentiment classification.

It also includes a **Streamlit app** for real-time sentiment prediction using the fine-tuned model.

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/selvintuscano/finetune-distilbert-lora-amazon-polarity.git
cd finetune-distilbert-lora-amazon-polarity
```

### 2️⃣ Install Dependencies
Make sure you have **Python 3.9+** installed.

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
cd sentiment_model
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.  
Type any product review to get a real-time **Positive** or **Negative** sentiment prediction!

---

## 📂 Project Structure
```
├── Amazon_Polarity.ipynb         # Notebook for training, tuning, evaluation
├── results_lr=...                # Training outputs for hyperparameter configs
├── sentiment_model/              # Fine-tuned model + Streamlit app
│   ├── app.py                    # Streamlit app script
│   ├── adapter_config.json       # LoRA adapter config
│   ├── adapter_model.safetensors # LoRA adapter weights
│   ├── tokenizer files           # Tokenizer configs
│   └── README.md
├── Technical Report              # Project documentation
├── Sentiment_Classification.pptx # Presentation slides
├── requirements.txt              # Dependencies list
└── LICENSE                       # MIT License
```

---

## 🛠️ Features
- ⚡ **Parameter-Efficient Fine-Tuning (PEFT)** using LoRA
- 🧠 Lightweight **DistilBERT** model
- 📊 Achieved **90.2% accuracy** on test set
- 🔧 Manual hyperparameter tuning
- 🖥️ **Streamlit App** for live sentiment predictions
- 📈 Evaluation includes Accuracy, F1-Score, and Confusion Matrix
- 🔍 Error analysis to identify model weaknesses

---

## 🧠 Model Details
- **Base Model:** `distilbert-base-uncased`
- **Fine-Tuning Method:** LoRA via Hugging Face **PEFT**
- **Dataset:** [Amazon Polarity](https://huggingface.co/datasets/amazon_polarity)  
  (Downsampled to 1,000 training and 1,000 test samples)
- **Trainable Parameters:** ~0.93% of total model size
- **Final Test Accuracy:** 90.2%

---

## 📊 Results Summary

| Metric     | Score  |
|------------|--------|
| Accuracy   | 90.2%  |
| Precision  | 0.90   |
| Recall     | 0.90   |
| F1-Score   | 0.90   |

- Balanced performance across both sentiment classes.
- Detailed evaluation available in the notebook, including confusion matrix and misclassified examples.

---

## 🖥️ Streamlit App Preview
_Easily predict sentiment by entering review text in a simple web interface._

Example:
```
Review: "Absolutely loved this product! Highly recommend."
Prediction: Positive
```

Launch locally with:
```bash
cd sentiment_model
streamlit run app.py
```

---

## 🤖 Python Inference Example
Use the fine-tuned model directly in your Python scripts:

```python
from app import predict_sentiment

review = "The product stopped working after a week. Very disappointed."
print(predict_sentiment(review))   # Output: Negative
```

---

## 📦 Requirements
Core libraries used in this project:

- `transformers`
- `datasets`
- `peft`
- `torch`
- `streamlit`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🎯 Future Enhancements
- 🚀 Deploy the Streamlit app on **Streamlit Cloud**
- 📊 Train on a larger dataset for improved generalization
- 🤖 Explore **Prompt-Tuning** and **AdapterFusion**
- 📝 Handle complex cases like sarcasm and mixed sentiments
- 🌐 Build an API endpoint using **FastAPI** for production use

---

## 📚 References
- [DistilBERT: Smaller, Faster, Cheaper](https://arxiv.org/abs/1910.01108)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
- [Amazon Polarity Dataset](https://huggingface.co/datasets/amazon_polarity)
- [Transformers by Hugging Face](https://huggingface.co/transformers)

---

## ⚖️ License
This project is licensed under the **MIT License**.  
© 2025 Selvin Tuscano

Feel free to use, modify, and distribute this project for personal or commercial purposes.

---
