# 🧠 Ethereum Fraud Detection using Machine Learning & Graph Neural Networks

### 📌 Project Overview
This project focuses on detecting fraudulent Ethereum addresses by analyzing transaction data.  
Two main approaches were explored:
1. **Traditional Machine Learning models** (Random Forest, XGBoost)
2. **Graph Neural Network (GNN)** based model to capture relationships between addresses

---

## 🚀 Objectives
- Analyze Ethereum transaction data to identify fraudulent activity.  
- Compare performance between traditional ML and GNN models.  
- Handle class imbalance and improve fraud recall.  

---

## 📂 Dataset
**Source:** [Ethereum Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)  
**File Used:** `transaction_dataset.csv`

**Key Details:**
- Each row represents an Ethereum address with transaction statistics.
- Target column: `FLAG` → `1 = Fraud`, `0 = Non-Fraud`
- Highly imbalanced dataset — only a small fraction of addresses are fraudulent.

---

## ⚙️ Steps & Methodology

### 🧹 1. Data Preprocessing
The raw dataset contained multiple unused or redundant fields (like indices and address IDs).  
Steps performed:
- **Dropped irrelevant columns:** such as `Unnamed: 0`, `Address`, and other identifiers that don’t affect prediction.  
- **Converted categorical columns** (e.g., account types) to numeric using Pandas category encoding for efficiency.  
- **Checked for missing values** visually using a heatmap and filled them using **median imputation**.  
- **Removed low-variance features** — columns where all values were constant were dropped since they add no learning value.  
- Verified final feature consistency using `.info()` and `.describe()` summaries.

---

### ⚖️ 2. Handling Imbalanced Data
Fraudulent transactions made up only a small percentage of total records.  
To ensure balanced learning:
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples for the minority class (`FLAG = 1`).  
- Verified class distribution after balancing with a pie chart.  
- This ensured models didn’t get biased toward the majority (non-fraud) class.

---

### 📊 3. Exploratory Data Analysis (EDA)
Performed to understand trends and relationships:
- **Correlation matrix heatmap:** to identify which numerical features were strongly correlated.  
- **Distribution plots:** to visualize differences between fraudulent and legitimate accounts.  
- **Feature variance check:** identified features with zero or near-zero variance for removal.
- **Pie charts and count plots:** highlighted severe class imbalance before resampling.

---

### 🧩 4. Feature Engineering
- Standardized numerical columns using **MinMaxScaler** to normalize feature ranges between 0 and 1.  
- Selected top relevant features by checking correlation with the target variable (`FLAG`).  
- Created derived metrics such as **average transaction value** and **transaction ratio** when applicable.

---

### 🤖 5. Baseline Machine Learning Models
Trained a set of standard models for comparison:
- **Logistic Regression** – quick linear baseline.  
- **Random Forest** – strong tree-based ensemble for feature-rich data.  
- **XGBoost** – gradient boosting method for accuracy and feature importance.

Each model was trained on the resampled (SMOTE-balanced) dataset and evaluated using:
- Accuracy  
- Precision  
- Recall (especially for fraud detection)  
- F1-score  
- ROC-AUC  

> 📈 *Random Forest achieved around 94% accuracy and 72% recall for fraud detection.*

---

### 🔗 6. Graph Construction for GNN
To better capture relationships between accounts:
- Represented each **Ethereum address as a node**.  
- Created **edges** between nodes where transactions occurred.  
- Node features included numerical transaction attributes (total ether sent, received, etc.).  
- Used **NetworkX** or **PyTorch Geometric** to build a graph object from the tabular data.

This structure allowed the model to learn **how connected addresses influence each other’s behavior**, improving fraud detection.

---

### 🧠 7. Graph Neural Network (GNN) Modeling
Implemented a **Graph Convolutional Network (GCN)** using **PyTorch Geometric**:

**Architecture:**
1. **Input Layer:** accepts node features.  
2. **GraphConv Layers:** aggregate neighborhood information to update node representations.  
3. **ReLU Activation + Dropout:** for regularization and non-linearity.  
4. **Output Layer:** predicts whether each address (node) is fraudulent or not.

**Training:**
- Optimizer: **Adam**  
- Loss: **CrossEntropyLoss**  
- Epochs: 50–100 (tuned experimentally)

This model learns **patterns of fraud propagation** — how fraudulent nodes are connected to others.

---

### 🧮 8. Evaluation & Comparison
After training both approaches, performance was compared using:
| Model | Accuracy | Recall (Fraud) | F1-Score |
|--------|-----------|----------------|-----------|
| Random Forest | 94% | 72% | 80% |
| XGBoost | 93% | 74% | 82% |
| GNN (GCN) | 91% | 86% | 88% |

**Observations:**
- The GNN slightly lowered accuracy (due to more balanced classification)  
- But significantly improved **recall** and **F1-score** for fraud cases  
- This means it **caught more fraudulent accounts** that traditional models missed.

---

## 💡 Key Takeaways
- Traditional ML models are fast and simple but limited to independent features.  
- GNN captures **contextual and relational signals** in blockchain data, improving fraud detection.  
- Handling imbalance and cleaning low-variance features are crucial for fair model training.  
- Real-world fraud often spreads through connected entities — GNNs can detect those patterns.

---

## 🧰 Tech Stack
- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn**
- **PyTorch & PyTorch Geometric**
- **SMOTE (imbalanced-learn)**

