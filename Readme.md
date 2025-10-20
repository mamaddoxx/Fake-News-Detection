# 📰 Fake News Detection – (FA-KES Dataset)

> Detection of Fake vs. Real news articles — following  
> *“Fake news detection: A hybrid CNN-RNN based deep learning approach”* (IJIM Data Insights, 2021).  

---

## 📚 Project Overview

This project reproduces and extends the hybrid **CNN → LSTM** model for fake-news detection proposed in the referenced paper.  
It also implements two **traditional baselines (TF-IDF + Linear SVC / Logistic Regression)** for comparison.

### Objectives
1. Detect fake vs. real news using the **FA-KES** dataset.  
2. Reproduce both **classical ML** and **deep learning** models described in the paper.  
3. Compare performance metrics (**Accuracy, Precision, Recall, F1**) and visualize training progress.  
4. Analyze why deep models outperform lexical baselines.

---

## 🧩 Dataset Description – FA-KES

| Column | Description |
|---------|--------------|
| `unit_id` | Unique ID for each article |
| `article_title` | News title |
| `article_content` | Main text body (used as input) |
| `source` | Original publisher |
| `date` | Publication date |
| `location` | Reported location |
| `labels` | Ground-truth label (`0 = Fake`, `1 = Real`) |

The dataset is pre-provided as **`FA-KES-Dataset.csv`** and automatically loaded by the script.

---

## 🧠 Model Summary

| Category | Model | Description |
|-----------|--------|-------------|
| **Baseline 1** | **TF-IDF + Linear SVC** | Extracts n-gram TF-IDF features; classifies via Support Vector Machine. |
| **Baseline 2** | **TF-IDF + Logistic Regression** | Probabilistic linear model using TF-IDF vectors. |
| **Deep Model 1** | **RNN (LSTM)** | Learns sequential dependencies across words. |
| **Deep Model 2** | **Hybrid CNN → MaxPool → LSTM** | Captures local phrase patterns (CNN) and temporal context (LSTM). |

### Motivation
- **Baselines** act as reference (“traditional ML”) and correspond to Section 4.1 / Table 2 of the paper.  
- **RNN** captures sentence-level context missed by TF-IDF.  
- **Hybrid CNN-RNN** combines local (CNN) and global (LSTM) semantics, achieving state-of-the-art F1 ≈ 0.8 on FA-KES.

---

## 🧰 Requirements

```bash
python >= 3.9
pandas
numpy
scikit-learn
matplotlib
tensorflow >= 2.10   # for deep models
