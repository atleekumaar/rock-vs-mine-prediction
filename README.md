 #  Rock vs Mine Prediction 

This project uses Machine Learning to predict whether a detected underwater object is a **Rock** or a **Mine** based on SONAR frequency data.

---

## Project Overview
Sonar (sound navigation and ranging) data is collected from different angles to detect object surfaces. This repository contains an end-to-end classification pipeline that uses Logistic Regression to classify underwater signals.

---

##  Tech Stack & Libraries
- **Language:** Python 3
- **Data Handling:** NumPy, Pandas
- **Machine Learning:** Scikit-learn (Logistic Regression)

---

##  Dataset Details
- **Source:** SONAR Dataset
- **Features:** 60 numerical sonar frequencies (ranging from 0.0 to 1.0)
- **Target Classes:**
  - `R` ➔ **Rock**
  - `M` ➔ **Mine**

---

##  Model Performance
- **Training Accuracy:** ~83.4%
- **Test Accuracy:** ~76.1%

---
## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/atleekumaar/rock-vs-mine-prediction.git](https://github.com/atleekumaar/rock-vs-mine-prediction.git)
cd rock-vs-mine-prediction

