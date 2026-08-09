 #  Rock vs Mine Prediction 

This project uses Machine Learning to predict whether a detected underwater object is a **Rock** or a **Mine** based on SONAR frequency data.
Link of web app ==> https://blank-app-faabjyd1lpg.streamlit.app/

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

```





                                        

     


### 2. Install dependencies

```bash
pip install -r requirements.txt

```

### 3. Launch the Streamlit App

```bash
streamlit run app.py

```

*(Replace `app.py` with your Streamlit entry point script if it uses a different name, e.g., `main.py`)*

---

## 💡 How to Use the Web App

1. Enter or select the 60 SONAR frequency values in the web interface.
2. Click the **Predict** button.
3. View the model's output predicting whether the object is a **Rock** or a **Mine**.
