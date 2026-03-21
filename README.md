# IoT Malware Detection – Data Preprocessing

## 📌 Overview

This module is responsible for preparing the **global cleaned dataset** used across all models in this project:

- Random Forest (baseline)
- LSTM (deep learning baseline)
- DRL agent (main objective)

The goal is to create a **neutral, reusable dataset** that is:
- clean
- consistent
- model-agnostic

---

## 🎯 Objectives

- Clean raw IoT network data
- Standardize formats and data types
- Handle missing and invalid values
- Normalize the target label
- Remove duplicates
- Produce a **single shared dataset** for all models

---

## 📊 Input Dataset

Raw dataset:
```
iot_dataset_raw.csv
```

Contains network traffic features such as:
- packet counts
- byte volumes
- protocol information
- HTTP / SSL / DNS attributes

---

## 🧹 Preprocessing Steps

The preprocessing pipeline is implemented in:
```
preprocess.py
```

---

### 1. Column Selection

Only relevant columns are kept:

- Network traffic:
  - `duration`, `src_bytes`, `dst_bytes`
  - `src_pkts`, `dst_pkts`
  - `src_ip_bytes`, `dst_ip_bytes`
- Protocol:
  - `proto`, `dns_qclass`, `dns_qtype`, `dns_rejected`
- SSL:
  - `ssl_version`, `ssl_cipher`, `ssl_resumed`
- HTTP:
  - `http_trans_depth`, `http_method`, `http_version`
  - `http_request_body_len`, `http_response_body_len`
- Metadata:
  - `src_ip`
  - `type`
- Target:
  - `label`

---

### 2. Missing Value Standardization

Common placeholders are converted to `NaN`:

- `" - "`, `"--"`, `"null"`, `"None"`, `"n/a"`, etc.

---

### 3. Target Cleaning

The `label` column is normalized into binary values:

- `0` → benign
- `1` → malicious

Supports multiple formats:
- `Benign / Malicious`
- `Normal / Attack`
- `0 / 1`
- etc.

---

### 4. Data Type Correction

- Numeric columns → converted using safe casting
- Categorical columns → kept as strings
- Invalid values → converted to `NaN`

---

### 5. Duplicate Removal

Exact duplicate rows are removed to avoid bias.

---

### 6. Missing Values Policy

- Rows are **NOT dropped** for missing features
- Only rows with missing `label` are removed

👉 Missing values are handled later in model pipelines

---

### 7. Constant Column Removal

Columns with no variation are removed automatically.

---

### 8. No Feature Engineering (Important)

This preprocessing step intentionally does NOT include:

- encoding (OneHot, LabelEncoder)
- scaling (StandardScaler, MinMaxScaler)
- normalization
- feature selection
- data balancing

👉 These operations are **model-specific** and handled later

---

## 📦 Output Dataset
```
iot_dataset_clean.csv
```

This dataset is:

- cleaned
- consistent
- ready for all models

---

## ⚠️ Design Philosophy

This preprocessing step is **shared across all models**.

### Why?

Different models require different preprocessing:

| Model | Needs |
|------|------|
| Random Forest | tabular numeric features |
| LSTM | sequential data |
| DRL | custom state representation |

👉 Therefore:

> The dataset is kept **neutral**, and each model performs its own transformations.

---

## 🚫 Important Constraints

- Do NOT modify this dataset per model
- Do NOT encode or scale here
- Do NOT drop columns here for specific models

---

📂 Folder Structure
```
Data/
│── preprocess.py
│── iot_dataset_raw.csv
│── iot_dataset_clean.csv
│── README.md
```

## ▶️ How to Run

```bash
cd Data
python preprocess.py
```




