# 📩 SMS Spam Classifier

## 📌 Project Overview

The **SMS Spam Classifier** is a machine learning–based application that automatically classifies incoming SMS messages as **Spam** or **Ham (Not Spam)**. The project demonstrates a complete **end-to-end NLP and Machine Learning workflow**, from raw text preprocessing to model deployment using a simple web interface.

This project is designed to be:

* Simple to understand
* Practical and real-world oriented
* Suitable for academic submissions
* Strong enough for an AI/ML portfolio on GitHub

---

## 🎯 Problem Statement

With the increasing number of promotional and fraudulent SMS messages, manual filtering has become inefficient. The goal of this project is to build an intelligent system that can automatically detect spam messages using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

---

## 🧠 Solution Approach

The SMS Spam Classifier uses a **Supervised Machine Learning** approach where the model is trained on labeled SMS data. Each message is processed using NLP techniques and then classified using a trained ML model.

The solution involves:

1. Cleaning and preprocessing raw SMS text
2. Converting text into numerical features
3. Training a classification model
4. Deploying the model using a web interface

---

## 🛠️ Technologies Used

### Programming Language

* **Python**

### Libraries & Frameworks

* `pandas` – data handling and preprocessing
* `numpy` – numerical operations
* `nltk` – text preprocessing
* `scikit-learn` – model training and evaluation
* `pickle` – model serialization
* `streamlit` – interactive web interface

---
---

## 📊 Dataset Description

The dataset consists of SMS messages labeled as:

* **Spam** – unwanted promotional or fraudulent messages
* **Ham** – legitimate messages

Each record contains:

* The SMS text
* The corresponding label (Spam/Ham)

---

## ⚙️ NLP Pipeline & Feature Engineering

The following NLP steps are applied to each SMS message:

1. Lowercasing the text
2. Removing punctuation and special characters
3. Tokenization
4. Stop-word removal
5. Stemming using Porter Stemmer
6. Vectorization using **TF-IDF Vectorizer**

This process converts raw text into numerical features that can be used by ML algorithms.

---

## 🤖 Machine Learning Model

* **Algorithm Used:** Naive Bayes (Multinomial)
* **Reason for Choice:**

  * Performs well on text classification tasks
  * Computationally efficient
  * Works effectively with TF-IDF features

The model is trained on preprocessed SMS data and evaluated using standard classification metrics.

---

## 📈 Model Evaluation

The classifier is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

These metrics ensure that the model performs well, especially in detecting spam messages without misclassifying genuine ones.

---

## 🖥️ Web Interface

The project uses **Streamlit** to provide a clean and interactive UI where:

* Users can enter an SMS message
* The system predicts whether the message is Spam or Ham
* Results are displayed instantly

This makes the project easy to demo and test.

---

## 🚀 How to Run the Project

Follow the steps below to run the project locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

### 5️⃣ Open in Browser

After the server starts, open:

```
http://localhost:8501
```

---

## 📌 Use Cases

* SMS spam detection systems
* Email or message filtering applications
* NLP and text classification learning projects
* ML model deployment demonstrations

---

## 📈 Future Enhancements

* Use advanced models (Logistic Regression, SVM)
* Add deep learning–based text classifiers
* Store prediction history
* Deploy on cloud platforms
* Improve UI with message statistics

---

## 🧪 Learning Outcomes

* Hands-on experience with NLP pipelines
* Understanding text vectorization techniques
* Training and evaluating ML classification models
* Deploying ML models using Streamlit

---

## 📜 Conclusion

The SMS Spam Classifier demonstrates how machine learning and NLP techniques can be applied to solve a real-world problem. The project covers the complete ML lifecycle, making it a strong addition to any AI/ML portfolio.

---

## 👤 Author

**Aaryan Shukla**
MSc Artificial Intelligence
AI/ML Engineer

---

> *Feel free to fork, modify, and enhance this project.*
