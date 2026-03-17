# 🛍️ NLP Product Classifier

> Benchmarking three NLP approaches — TF-IDF + Dense NN → BiLSTM + GloVe → DistilBERT — on a 3-class e-commerce product review classification task, improving accuracy from ~91% to ~97% while analyzing the accuracy/latency tradeoff for production deployment.

---

## 📌 Problem Statement

An e-commerce platform collects customer product reviews. The goal is to automatically classify each review into one of three product categories:

| Label | Category |
|-------|----------|
| 0 | Movie DVD |
| 1 | Electronics |
| 2 | Kitchen Appliances |

**Input**: Raw product review text (`desc`)  
**Output**: Product category label (0 / 1 / 2)  
**Metric**: Accuracy

---

## 🗂️ Project Structure

```
nlp-product-classifier/
│
├── data/                        # Raw and processed data
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── submissions.csv          # Generated predictions
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 02_baseline_tfidf.ipynb  # Approach 1: TF-IDF + Dense NN
│   ├── 03_lstm_glove.ipynb      # Approach 2: BiLSTM + GloVe embeddings
│   └── 04_bert_finetune.ipynb   # Approach 3: Fine-tuned DistilBERT
│
├── src/
│   ├── preprocess.py            # Text cleaning & vectorization utilities
│   ├── models.py                # Model definitions (NN, LSTM, BERT)
│   └── evaluate.py              # Metrics, plots, confusion matrix
│
├── results/
│   └── model_comparison.png     # Accuracy vs latency comparison chart
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Approaches & Results

| # | Approach | Accuracy | Inference Latency | Best For |
|---|----------|----------|-------------------|----------|
| 1 | TF-IDF + Dense Neural Network | ~91% | ⚡ Fast (~2ms) | Baseline, resource-constrained |
| 2 | BiLSTM + GloVe Embeddings | ~94% | 🔶 Medium (~15ms) | Balanced accuracy/speed |
| 3 | Fine-tuned DistilBERT | ~97% | 🔴 Slow (~80ms) | Highest accuracy |

---

## 🧠 Technical Deep Dive

### Approach 1 — TF-IDF + Dense Neural Network
- **Preprocessing**: Lowercasing, punctuation removal, TF-IDF vectorization (`max_features=10000`, `ngram_range=(1,2)`, `sublinear_tf=True`)
- **Architecture**: `Dense(512) → BN → Dropout(0.4) → Dense(256) → BN → Dropout(0.3) → Dense(128) → Dense(3, softmax)`
- **Anti-overfitting**: BatchNormalization + Dropout + EarlyStopping with `restore_best_weights=True`
- **Why `sublinear_tf`?** Log-scales term frequency so high-frequency filler words don't dominate over rare but informative product terms

### Approach 2 — BiLSTM + GloVe Embeddings
- **Embeddings**: Pre-trained GloVe 100d vectors (6B tokens)
- **Architecture**: `Embedding(pretrained) → SpatialDropout1D → BiLSTM(128) → GlobalMaxPool → Dense(64) → Dense(3, softmax)`
- **Why BiLSTM?** Captures sequential context in both directions — "not good" vs "good" are semantically opposite but look similar to bag-of-words models

### Approach 3 — Fine-tuned DistilBERT
- **Model**: `distilbert-base-uncased` from HuggingFace Transformers
- **Strategy**: Freeze base layers for first 2 epochs, unfreeze top 2 transformer blocks for fine-tuning
- **Why DistilBERT over BERT?** 40% smaller, 60% faster, retains 97% of BERT's performance — ideal production tradeoff

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/pranayhedau007/nlp-product-classifier.git
cd nlp-product-classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Running the Notebooks

Run notebooks in order:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

Each notebook is self-contained and includes markdown cells explaining every design decision.

---

## 🔑 Key Design Decisions & Interview Talking Points

1. **`stratify=y_train` in train/val split** — Preserves class distribution, critical for imbalanced datasets
2. **`sparse_categorical_crossentropy` vs `categorical_crossentropy`** — Integer labels vs one-hot; using the wrong one is a silent bug
3. **`restore_best_weights=True` in EarlyStopping** — Without this you get last-epoch weights, not best-epoch weights
4. **Bigrams in TF-IDF** — "kitchen appliance" carries far more signal than "kitchen" or "appliance" alone
5. **Frozen → unfrozen fine-tuning strategy** — Prevents catastrophic forgetting of BERT's pre-trained representations

---

## 📁 Data Format

**train.csv / test.csv**
```
desc,label
"Great picture quality, loved the movie",0
"Excellent sound, fast charging",1
"Easy to clean, heats evenly",2
```

**submissions.csv** (output)
```
desc,label
"Fast delivery and good packaging",1
...
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)

- **Deep Learning**: TensorFlow / Keras
- **NLP**: scikit-learn (TF-IDF), GloVe, HuggingFace Transformers
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Notebooks**: Jupyter

---

## 👤 Author

**Pranay Hedau**  
MS Computer Science @ UC Irvine  
[LinkedIn](https://linkedin.com/in/pranayhedau) · [GitHub](https://github.com/pranayhedau007) · [YouTube](https://youtube.com/@pranayhedau)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
