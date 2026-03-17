# 📄 Resume Classifier — ML Web App

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-22C55E?style=flat-square)

> End-to-end NLP pipeline that reads raw resume text and instantly predicts the candidate's job category — trained on 9 ML classifiers, deployed live on Streamlit Cloud.

🚀 **[Open Live App](https://customerchurntelecom-qlhhzokxrbhccgc5numfni.streamlit.app/)** &nbsp;|&nbsp; 📄 **[Project Report](#)** &nbsp;|&nbsp; ⭐ **Star this repo if you found it helpful!**

---

## 📸 App Screenshots

## 📸 App Screenshots

| 🔍 Classify Resume | 📊 Model Performance |
|:---:|:---:|
| ![Classify]([screenshot/app.png](https://github.com/gaganasindhu-ml/resume-_classification-/blob/main/screenshot/app.png)) | ![Models]([screenshot/model_comparision.png](https://github.com/gaganasindhu-ml/resume-_classification-/blob/main/screenshot/model_comparision.png)) |

| 📈 Accuracy Chart | 🔥 Confusion Matrix |
|:---:|:---:|
| (https://github.com/gaganasindhu-ml/resume-_classification-/blob/main/screenshot/accuracy_comparision.png?raw=true))| !(https://github.com/gaganasindhu-ml/resume-_classification-/blob/main/screenshot/confusion%20matrix.png) |

---

## 🧠 Project Overview

This project automates resume screening using Machine Learning. It reads raw resume text and classifies it into one of **4 technical job categories** with a confidence percentage.

### 📂 Dataset — 79 Resumes, 4 Categories

| Category | Count | Domain |
|---|:---:|---|
| 🏢 PeopleSoft | 20 | ERP Platform Specialist |
| ⚛️ React JS Developer | 24 | Frontend Web Developer |
| 🗄️ SQL Developer | 14 | Database & Analytics Engineer |
| ☁️ Workday | 21 | HCM Cloud Consultant |

---

## 🤖 Model Results

| Model | Train Acc | Test Acc | F1 Score |
|---|:---:|:---:|:---:|
| ✅ Random Forest ⭐ | 100% | **100%** | 100% |
| ✅ Decision Tree | 100% | **100%** | 100% |
| ✅ Bagging | 100% | **100%** | 100% |
| ✅ Gradient Boost | 100% | **100%** | 100% |
| ✅ SVM | 100% | **100%** | 100% |
| ✅ KNN | 100% | **100%** | 100% |
| ⚡ Logistic Reg. | 100% | 95% | 94.9% |
| ⚡ Naive Bayes | 100% | 90% | 90.2% |
| ❌ AdaBoost | 42.4% | 45% | 31.6% |

**Best Model: Random Forest (100% Test Accuracy)**

---

## ⚙️ How It Works

```
Raw Resume Text
      │
      ▼
TF-IDF Vectorization  (sublinear_tf=True, 3630 features)
      │
      ▼
Random Forest Classifier
      │
      ▼
Predicted Category + Confidence %
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Streamlit | 1.32 | Web app framework |
| scikit-learn | 1.3 | ML models + TF-IDF |
| Plotly | 5.18 | Interactive charts |
| Pandas | 2.1 | Data processing |
| NumPy | 1.26 | Numerical operations |

---

## ✨ App Features

**Tab 1 — 🔍 Classify Resume**
- Paste any resume text → instant prediction
- Confidence % with probability bar chart
- Domain keyword highlights
- 4 quick-sample buttons

**Tab 2 — 📊 Model Performance**
- Comparison table for all 9 models
- Test Accuracy bar chart
- Confusion Matrix heatmap
- Classification Report

**Tab 3 — 📈 Data Explorer**
- Category distribution charts
- Word count histogram

**Tab 4 — 📋 Dataset Viewer**
- Filter + search resumes
- CSV download

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/resume-classification.git
cd resume-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501` 🎉

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in → **New app** → select repo
4. Main file: `app.py` → **Deploy**

---

## 📁 Project Structure

```
resume-classification/
│
├── app.py                         ← Streamlit app
├── Cleaned_Resumes.csv            ← Dataset (79 resumes)
├── requirements.txt               ← Dependencies
├── README.md                      ← This file
│
├── notebooks/                     ← Jupyter notebooks
│   ├── classification_eda.ipynb
│   └── model_building1.ipynb
│
├── data/                          ← Raw & processed data
│
├── screenshots/                   ← App screenshots
│   ├── classify.png
│   ├── models.png
│   ├── accuracy_chart.png
│   └── confusion_matrix.png
│
└── images/                        ← README display images
```

---

## 👤 Author

**Gaganasindhu**
- GitHub: [@gaganasindhu-ml](https://github.com/gaganasindhu-ml)

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

⭐ **If this project helped you, please give it a star!**
