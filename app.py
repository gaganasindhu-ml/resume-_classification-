import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                              precision_score, recall_score, classification_report)
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────
CATEGORY_META = {
    "PeopleSoft": {
        "color": "#FF6B35",
        "icon": "🏢",
        "desc": "ERP Platform Specialist",
        "keywords": ["peoplesoft","tuxedo","weblogic","hcm","fscm","pia","psadmin","pum","peopletools","oracle"]
    },
    "React JS Developer": {
        "color": "#61DAFB",
        "icon": "⚛️",
        "desc": "Frontend Web Developer",
        "keywords": ["react","jsx","redux","hooks","javascript","typescript","node","webpack","next","component"]
    },
    "SQL Developer": {
        "color": "#00C853",
        "icon": "🗄️",
        "desc": "Database Engineer",
        "keywords": ["sql","database","query","stored procedure","mysql","postgresql","etl","tableau","ssrs","oracle"]
    },
    "Workday": {
        "color": "#A78BFA",
        "icon": "☁️",
        "desc": "HCM Cloud Consultant",
        "keywords": ["workday","hcm","payroll","integration","studio","eis","bre","absence","compensation","recruit"]
    }
}

# FIXED HERE
PLOTLY_BASE = dict(
    paper_bgcolor='#0d1117',
    plot_bgcolor='#161b22',
    font=dict(color='#c9d1d9', family='Space Grotesk')
)

# ── Train Models ─────────────────────────────────────
@st.cache_resource
def train_models():

    df = pd.read_csv(os.path.join(BASE_DIR, "Cleaned_Resumes.csv"))

    x = df["Resume_Details"].values
    y = df["Category"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=45, test_size=0.25, stratify=y
    )

    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words='english')

    x_tr = tfidf.fit_transform(x_train)
    x_te = tfidf.transform(x_test)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Logistic Reg.": LogisticRegression(max_iter=1000),
        "Bagging": BaggingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boost": GradientBoostingClassifier(),
        "Naive Bayes": MultinomialNB()
    }

    trained = {}

    for name, model in models.items():

        model.fit(x_tr, y_train)
        y_pred = model.predict(x_te)

        trained[name] = {
            "model": model,
            "train_acc": round(model.score(x_tr, y_train)*100,1),
            "test_acc": round(accuracy_score(y_test,y_pred)*100,1),
            "precision": round(precision_score(y_test,y_pred,average='weighted')*100,1),
            "recall": round(recall_score(y_test,y_pred,average='weighted')*100,1),
            "f1": round(f1_score(y_test,y_pred,average='weighted')*100,1),
            "y_pred": y_pred,
            "report": classification_report(y_test,y_pred,output_dict=True)
        }

    return tfidf, trained, df, y_test


tfidf, trained_models, df, y_test = train_models()

# ── Sidebar ─────────────────────────────────────
with st.sidebar:

    st.title("📄 Resume Classifier")

    selected_model = st.selectbox(
        "Select Model",
        list(trained_models.keys())
    )

    st.markdown("---")

    st.write("Dataset")

    st.write("Resumes:",len(df))
    st.write("Categories:",df["Category"].nunique())

    acc = trained_models[selected_model]["test_acc"]

    st.metric("Model Accuracy",f"{acc}%")


# ── Header ─────────────────────────────────────
st.title("📄 Resume Classifier")

m1,m2,m3 = st.columns(3)

m1.metric("Active Model",selected_model)
m2.metric("Accuracy",f"{trained_models[selected_model]['test_acc']}%")
m3.metric("Resumes",len(df))

st.markdown("---")

# ── Resume Input ────────────────────────────────
resume_text = st.text_area("Paste Resume Text",height=250)

if st.button("Classify Resume"):

    model_obj = trained_models[selected_model]["model"]

    vec = tfidf.transform([resume_text])

    pred = model_obj.predict(vec)[0]

    meta = CATEGORY_META[pred]

    st.success(f"Prediction: {pred}")

    if hasattr(model_obj,"predict_proba"):

        probs = model_obj.predict_proba(vec)[0]

        prob_dict = dict(zip(model_obj.classes_,probs))

        sorted_probs = sorted(prob_dict.items(),key=lambda x:x[1],reverse=True)

        cats = [c for c,_ in sorted_probs]
        vals = [round(v*100,1) for _,v in sorted_probs]
        colors = [CATEGORY_META[c]["color"] for c in cats]

        fig = go.Figure(go.Bar(

            x=vals,
            y=cats,
            orientation='h',
            marker_color=colors,
            text=[f"{v}%" for v in vals],
            textposition="outside"

        ))

        fig.update_layout(

            **PLOTLY_BASE,
            height=200,
            title="Category Probabilities",
            xaxis=dict(range=[0,118]),
            margin=dict(l=10,r=60,t=35,b=10)

        )

        st.plotly_chart(fig,use_container_width=True)

# ── Model Performance ───────────────────────────
st.header("Model Performance")

rows = []

for name,data in trained_models.items():

    rows.append({
        "Model":name,
        "Train Accuracy":data["train_acc"],
        "Test Accuracy":data["test_acc"],
        "F1 Score":data["f1"]
    })

perf_df = pd.DataFrame(rows)

st.dataframe(perf_df)

fig = px.bar(

    perf_df,
    x="Model",
    y="Test Accuracy",
    title="Model Accuracy Comparison"

)

fig.update_layout(**PLOTLY_BASE)

st.plotly_chart(fig,use_container_width=True)

# ── Confusion Matrix ───────────────────────────
st.header("Confusion Matrix")

labels = sorted(df["Category"].unique())

cm = confusion_matrix(y_test,trained_models[selected_model]["y_pred"],labels=labels)

fig = go.Figure(go.Heatmap(

    z=cm,
    x=labels,
    y=labels,
    text=cm,
    texttemplate="%{text}",
    showscale=False

))

fig.update_layout(**PLOTLY_BASE)

st.plotly_chart(fig,use_container_width=True)
