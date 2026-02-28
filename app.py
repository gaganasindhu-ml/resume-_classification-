import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(97,218,251,0.1);
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #61DAFB !important; }

    /* Main bg */
    .stApp { background: #0d1117; }
    .main .block-container { padding: 2rem 2.5rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px !important;
    }
    [data-testid="metric-container"] label { color: #8b949e !important; font-size: 12px !important; letter-spacing: 2px; text-transform: uppercase; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #61DAFB !important; font-size: 28px !important; font-weight: 700 !important; }

    /* Headers */
    h1 { color: #ffffff !important; font-weight: 700 !important; }
    h2, h3 { color: #c9d1d9 !important; }

    /* Tabs */
    [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-baseweb="tab"] {
        color: #8b949e !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        letter-spacing: 1px !important;
    }
    [aria-selected="true"] {
        background: rgba(97,218,251,0.1) !important;
        color: #61DAFB !important;
        border-radius: 8px !important;
    }

    /* Text area */
    textarea {
        background: rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #c9d1d9 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(97,218,251,0.15), rgba(167,139,250,0.15)) !important;
        border: 1px solid rgba(97,218,251,0.4) !important;
        border-radius: 10px !important;
        color: #61DAFB !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(97,218,251,0.25), rgba(167,139,250,0.25)) !important;
        border-color: #61DAFB !important;
        transform: translateY(-1px) !important;
    }

    /* Selectbox */
    [data-baseweb="select"] {
        background: rgba(0,0,0,0.3) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* Result box styles */
    .result-box {
        background: rgba(97,218,251,0.05);
        border: 1px solid rgba(97,218,251,0.3);
        border-radius: 14px;
        padding: 24px;
        margin: 12px 0;
        text-align: center;
    }
    .category-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-top: 8px;
    }
    .info-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .keyword-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 11px;
        margin: 3px;
        font-family: 'JetBrains Mono', monospace;
    }
    .section-header {
        color: #8b949e;
        font-size: 11px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# ─── Constants ──────────────────────────────────────────────────────────────
CATEGORY_META = {
    "PeopleSoft": {
        "color": "#FF6B35", "icon": "🏢", "desc": "ERP Platform Specialist",
        "keywords": ["peoplesoft","tuxedo","weblogic","hcm","fscm","pia","psadmin","pum","peopletools","oracle"]
    },
    "React JS Developer": {
        "color": "#61DAFB", "icon": "⚛️", "desc": "Frontend Web Developer",
        "keywords": ["react","jsx","redux","hooks","javascript","typescript","node","webpack","next","component"]
    },
    "SQL Developer": {
        "color": "#00C853", "icon": "🗄️", "desc": "Database & Analytics Engineer",
        "keywords": ["sql","database","query","stored procedure","mysql","postgresql","etl","tableau","ssrs","oracle"]
    },
    "Workday": {
        "color": "#A78BFA", "icon": "☁️", "desc": "HCM Cloud Consultant",
        "keywords": ["workday","hcm","payroll","integration","studio","eis","bre","absence","compensation","recruit"]
    }
}

MODEL_RESULTS_DATA = {
    "Models": ["KNN", "DecisionTree", "RandomForest", "SVM", "Logistic Regression",
               "Bagging", "AdaBoost", "Gradient Boosting", "Naive Bayes"],
    "Train_Accuracy": [0.644, 1.0, 1.0, 1.0, 1.0, 1.0, 0.424, 1.0, 1.0],
    "Test_Accuracy":  [0.60,  1.0, 1.0, 0.95, 0.95, 1.0, 0.45, 1.0, 1.0],
    "Precision":      [0.48,  1.0, 1.0, 0.96, 0.96, 1.0, 0.28, 1.0, 1.0],
    "Recall":         [0.55,  1.0, 1.0, 0.95, 0.95, 1.0, 0.50, 1.0, 1.0],
    "F1_Score":       [0.47,  1.0, 1.0, 0.95, 0.95, 1.0, 0.34, 1.0, 1.0],
}

SAMPLES = {
    "🏢 PeopleSoft": "Experienced PeopleSoft HCM Administrator with 5 years managing PIA environments. Skilled in TuxedO, WebLogic, psadmin, PUM, Oracle database administration, and peopletools upgrades. Worked with HCM, FSCM modules in production support.",
    "⚛️ React JS Dev": "React JS Developer with expertise in Redux, hooks, context API, TypeScript, and Next.js. Built scalable frontend applications using component-based architecture, webpack, and modern JavaScript ES6+ for e-commerce platforms.",
    "🗄️ SQL Developer": "SQL Developer with 4 years experience writing complex queries, stored procedures, and ETL pipelines. Proficient in MySQL, PostgreSQL, Oracle optimization, Tableau dashboards, and SSRS reporting for financial data.",
    "☁️ Workday": "Workday HCM Consultant with Workday Studio integration experience. Implemented EIS integrations, BRE configuration, absence management, compensation, payroll, and recruitment modules for enterprise Fortune 500 clients."
}


# ─── Model Training (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models():
    df = pd.read_csv("Cleaned_Resumes.csv")
    encoder = LabelEncoder()
    df["Label"] = encoder.fit_transform(df["Category"])

    x = df["Resume_Details"].values
    y = df["Category"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=45, test_size=0.25, stratify=y
    )

    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words='english')
    x_train_vec = tfidf.fit_transform(x_train)
    x_test_vec  = tfidf.transform(x_test)

    models = {
        "KNN":              KNeighborsClassifier(n_neighbors=3),
        "Decision Tree":    DecisionTreeClassifier(random_state=42),
        "Random Forest":    RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":              SVC(kernel='linear', probability=True, random_state=42),
        "Logistic Reg.":    LogisticRegression(max_iter=1000, random_state=42),
        "Bagging":          BaggingClassifier(random_state=42),
        "AdaBoost":         AdaBoostClassifier(random_state=42),
        "Gradient Boost":   GradientBoostingClassifier(random_state=42),
        "Naive Bayes":      MultinomialNB(),
    }

    trained = {}
    for name, model in models.items():
        model.fit(x_train_vec, y_train)
        y_pred = model.predict(x_test_vec)
        trained[name] = {
            "model": model,
            "train_acc": round(model.score(x_train_vec, y_train) * 100, 1),
            "test_acc":  round(accuracy_score(y_test, y_pred) * 100, 1),
            "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 1),
            "recall":    round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 1),
            "f1":        round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 1),
            "y_pred":    y_pred,
            "report":    classification_report(y_test, y_pred, output_dict=True),
        }

    return tfidf, trained, encoder, df, x_test_vec, y_test


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Resume Classifier")
    st.markdown("*ML-Powered Talent Intelligence*")
    st.markdown("---")

    st.markdown("### 🤖 Select Model")
    selected_model = st.selectbox(
        "Classifier",
        ["Random Forest", "Decision Tree", "Bagging", "Gradient Boost",
         "Naive Bayes", "SVM", "Logistic Reg.", "KNN", "AdaBoost"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown("""
    <div style="font-size:13px; line-height:2; color:#8b949e;">
    📁 <b style="color:#c9d1d9">79</b> Resumes<br>
    🏷️ <b style="color:#c9d1d9">4</b> Categories<br>
    🔠 <b style="color:#c9d1d9">TF-IDF</b> Vectorizer<br>
    ✂️ <b style="color:#c9d1d9">75/25</b> Train/Test Split
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏷️ Categories")
    for cat, meta in CATEGORY_META.items():
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin:6px 0; padding:8px 10px;
             background:rgba(255,255,255,0.02); border-radius:8px; border-left:3px solid {meta['color']};">
            <span>{meta['icon']}</span>
            <div>
                <div style="font-size:12px; color:#c9d1d9; font-weight:600;">{cat}</div>
                <div style="font-size:10px; color:#8b949e;">{meta['desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:10px; color:#484f58; text-align:center;'>Built with Streamlit · scikit-learn</div>", unsafe_allow_html=True)


# ─── Load models ─────────────────────────────────────────────────────────────
with st.spinner("🔄 Training models on dataset..."):
    tfidf, trained_models, encoder, df, x_test_vec, y_test = train_models()

# ─── Header ──────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <h1 style="margin:0; font-size:2.2rem;">
        📄 Resume <span style="color:#61DAFB;">Classifier</span>
    </h1>
    <p style="color:#8b949e; margin:4px 0 20px; font-size:14px; letter-spacing:2px;">
        ML-POWERED TALENT INTELLIGENCE SYSTEM
    </p>
    """, unsafe_allow_html=True)
with col_h2:
    active_model_data = trained_models[selected_model]
    st.metric("Active Model", selected_model)
    st.metric("Test Accuracy", f"{active_model_data['test_acc']}%")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify Resume", "📊 Model Performance", "📈 Data Explorer", "📋 Dataset"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">INPUT RESUME TEXT</div>', unsafe_allow_html=True)

        # Sample buttons
        st.markdown("**Quick Samples:**")
        sample_cols = st.columns(4)
        for i, (label, text) in enumerate(SAMPLES.items()):
            with sample_cols[i]:
                if st.button(label, key=f"sample_{i}"):
                    st.session_state["resume_input"] = text

        resume_text = st.text_area(
            "Resume Text",
            value=st.session_state.get("resume_input", ""),
            height=280,
            placeholder="Paste resume content here...\n\nTip: Use the Quick Samples above to try it instantly.",
            label_visibility="collapsed",
            key="resume_input"
        )

        char_count = len(resume_text)
        word_count = len(resume_text.split()) if resume_text.strip() else 0
        st.markdown(f"<div style='font-size:11px; color:#484f58; text-align:right;'>{word_count} words · {char_count} chars</div>", unsafe_allow_html=True)

        classify_btn = st.button("⚡ Classify Resume", disabled=not resume_text.strip())

    with col_right:
        st.markdown('<div class="section-header">PREDICTION RESULT</div>', unsafe_allow_html=True)

        if not resume_text.strip():
            st.markdown("""
            <div style="height:340px; display:flex; flex-direction:column; align-items:center;
                        justify-content:center; border:1px dashed rgba(255,255,255,0.08);
                        border-radius:14px; color:#484f58; text-align:center; gap:12px;">
                <div style="font-size:48px; opacity:0.3;">⚡</div>
                <div style="font-size:13px;">Paste resume text and click Classify</div>
                <div style="font-size:11px; opacity:0.6;">Or use the Quick Samples on the left</div>
            </div>
            """, unsafe_allow_html=True)

        elif classify_btn or st.session_state.get("last_classified") == resume_text:
            st.session_state["last_classified"] = resume_text
            model_obj = trained_models[selected_model]["model"]

            vec = tfidf.transform([resume_text])
            prediction = model_obj.predict(vec)[0]
            meta = CATEGORY_META[prediction]

            # Probabilities if available
            if hasattr(model_obj, "predict_proba"):
                probs = model_obj.predict_proba(vec)[0]
                classes = model_obj.classes_
                prob_dict = dict(zip(classes, probs))
                confidence = prob_dict[prediction] * 100
            else:
                prob_dict = {prediction: 1.0}
                confidence = 100.0

            # Main result
            st.markdown(f"""
            <div style="background:{meta['color']}10; border:1px solid {meta['color']}50;
                        border-radius:14px; padding:24px; text-align:center; margin-bottom:16px;
                        position:relative; overflow:hidden;">
                <div style="position:absolute; top:0; left:0; right:0; height:3px;
                     background:linear-gradient(90deg, transparent, {meta['color']}, transparent);"></div>
                <div style="font-size:42px; margin-bottom:8px;">{meta['icon']}</div>
                <div style="font-size:11px; color:#8b949e; letter-spacing:3px; text-transform:uppercase; margin-bottom:8px;">
                    PREDICTED CATEGORY
                </div>
                <div style="font-size:26px; font-weight:700; color:{meta['color']}; margin-bottom:4px;">
                    {prediction}
                </div>
                <div style="font-size:12px; color:#8b949e; margin-bottom:20px;">{meta['desc']}</div>
                <div style="font-size:36px; font-weight:800; color:white;">{confidence:.1f}%</div>
                <div style="font-size:11px; color:#8b949e; letter-spacing:2px;">CONFIDENCE</div>
                <div style="margin-top:12px; background:rgba(0,0,0,0.2); border-radius:4px; height:6px;">
                    <div style="width:{confidence}%; height:100%; background:{meta['color']};
                         border-radius:4px; box-shadow:0 0 10px {meta['color']}80;"></div>
                </div>
                <div style="margin-top:8px; font-size:11px; color:#484f58;">
                    Using: <span style="color:#61DAFB;">{selected_model}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability breakdown
            if hasattr(model_obj, "predict_proba"):
                st.markdown('<div class="section-header" style="margin-top:8px;">CATEGORY PROBABILITIES</div>', unsafe_allow_html=True)
                for cat in sorted(prob_dict, key=prob_dict.get, reverse=True):
                    p = prob_dict[cat] * 100
                    m = CATEGORY_META.get(cat, {"color": "#666", "icon": "📄"})
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:10px; margin:6px 0;">
                        <span style="width:20px;">{m['icon']}</span>
                        <span style="width:130px; font-size:12px; color:#c9d1d9;">{cat}</span>
                        <div style="flex:1; background:rgba(255,255,255,0.05); border-radius:3px; height:6px;">
                            <div style="width:{p}%; height:100%; background:{m['color']}; border-radius:3px;"></div>
                        </div>
                        <span style="width:40px; text-align:right; font-size:12px; color:{m['color']}; font-weight:600;">
                            {p:.0f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            # Keywords found
            found_kw = [kw for kw in meta["keywords"] if kw in resume_text.lower()]
            if found_kw:
                st.markdown('<div class="section-header" style="margin-top:12px;">MATCHING KEYWORDS</div>', unsafe_allow_html=True)
                chips_html = "".join([
                    f'<span class="keyword-chip" style="background:{meta["color"]}15; border:1px solid {meta["color"]}30; color:{meta["color"]};">{kw}</span>'
                    for kw in found_kw
                ])
                st.markdown(chips_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Comparison")
    st.markdown("<p style='color:#8b949e; font-size:13px;'>All models trained on TF-IDF vectorized features, 75/25 stratified split</p>", unsafe_allow_html=True)

    # Build real metrics from trained models
    rows = []
    for name, data in trained_models.items():
        rows.append({
            "Model": name,
            "Train Acc %": data["train_acc"],
            "Test Acc %":  data["test_acc"],
            "Precision %": data["precision"],
            "Recall %":    data["recall"],
            "F1 Score %":  data["f1"],
        })
    perf_df = pd.DataFrame(rows).sort_values("Test Acc %", ascending=False)

    # Highlight table
    def highlight_perf(val):
        if isinstance(val, (int, float)):
            if val == 100.0:
                return "background: rgba(0,200,83,0.15); color: #00C853; font-weight:700;"
            elif val >= 90:
                return "background: rgba(97,218,251,0.08); color: #61DAFB;"
            elif val < 60:
                return "background: rgba(255,107,53,0.1); color: #FF6B35;"
        return "color: #c9d1d9;"

    styled = perf_df.style.applymap(
        highlight_perf, subset=["Train Acc %","Test Acc %","Precision %","Recall %","F1 Score %"]
    ).format("{:.1f}", subset=["Train Acc %","Test Acc %","Precision %","Recall %","F1 Score %"])

    st.dataframe(styled, use_container_width=True, height=380)

    # Charts
    st.markdown("---")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("**Test Accuracy by Model**")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')

        colors = ["#00C853" if v == 100 else "#61DAFB" if v >= 90 else "#FF6B35"
                  for v in perf_df["Test Acc %"]]
        bars = ax.barh(perf_df["Model"], perf_df["Test Acc %"], color=colors, height=0.6, alpha=0.85)

        for bar, val in zip(bars, perf_df["Test Acc %"]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val}%", va='center', ha='left', color='white', fontsize=10, fontweight='bold')

        ax.set_xlim(0, 112)
        ax.set_xlabel("Accuracy (%)", color='#8b949e', fontsize=11)
        ax.tick_params(colors='#c9d1d9', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('rgba(255,255,255,0.1)')
        ax.grid(axis='x', color='rgba(255,255,255,0.05)', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_c2:
        st.markdown("**F1 Score vs Test Accuracy**")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#161b22')

        x_pos = np.arange(len(perf_df))
        w = 0.35
        ax2.bar(x_pos - w/2, perf_df["Test Acc %"], width=w, label='Test Acc', color='#61DAFB', alpha=0.8)
        ax2.bar(x_pos + w/2, perf_df["F1 Score %"], width=w, label='F1 Score', color='#A78BFA', alpha=0.8)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(perf_df["Model"], rotation=45, ha='right', fontsize=9, color='#c9d1d9')
        ax2.tick_params(colors='#c9d1d9')
        for spine in ax2.spines.values():
            spine.set_color('rgba(255,255,255,0.1)')
        ax2.legend(facecolor='#161b22', labelcolor='white', fontsize=10)
        ax2.set_ylim(0, 115)
        ax2.grid(axis='y', color='rgba(255,255,255,0.05)', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Confusion matrix for selected model
    st.markdown("---")
    st.markdown(f"**Confusion Matrix — {selected_model}**")
    y_pred = trained_models[selected_model]["y_pred"]
    cm = confusion_matrix(y_test, y_pred, labels=sorted(df["Category"].unique()))
    labels = sorted(df["Category"].unique())

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    fig3.patch.set_facecolor('#0d1117')
    ax3.set_facecolor('#0d1117')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax3, linewidths=0.5, linecolor='#1c2128',
                cbar_kws={'shrink': 0.8})
    ax3.set_xlabel("Predicted", color='#8b949e', fontsize=12)
    ax3.set_ylabel("Actual", color='#8b949e', fontsize=12)
    ax3.tick_params(colors='#c9d1d9', labelsize=10, rotation=30)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Classification report
    st.markdown(f"**Classification Report — {selected_model}**")
    report = trained_models[selected_model]["report"]
    report_rows = []
    for label in labels:
        if label in report:
            r = report[label]
            report_rows.append({
                "Category": label,
                "Precision": f"{r['precision']*100:.1f}%",
                "Recall":    f"{r['recall']*100:.1f}%",
                "F1-Score":  f"{r['f1-score']*100:.1f}%",
                "Support":   int(r['support'])
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Dataset Explorer")

    # Summary metrics
    cat_counts = df["Category"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Resumes", len(df))
    m2.metric("Categories", df["Category"].nunique())
    m3.metric("Most Common", cat_counts.index[0])
    m4.metric("Least Common", cat_counts.index[-1])

    st.markdown("---")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**Category Distribution**")
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        fig4.patch.set_facecolor('#0d1117')
        ax4.set_facecolor('#0d1117')
        colors_pie = ["#FF6B35", "#61DAFB", "#00C853", "#A78BFA"]
        wedges, texts, autotexts = ax4.pie(
            cat_counts.values,
            labels=cat_counts.index,
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90,
            wedgeprops=dict(width=0.6, edgecolor='#0d1117', linewidth=2),
            pctdistance=0.75,
            labeldistance=1.1
        )
        for t in texts: t.set_color('#c9d1d9'); t.set_fontsize(11)
        for at in autotexts: at.set_color('white'); at.set_fontsize(10); at.set_fontweight('bold')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    with col_d2:
        st.markdown("**Resume Count per Category**")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        fig5.patch.set_facecolor('#0d1117')
        ax5.set_facecolor('#161b22')
        bar_colors = ["#FF6B35", "#61DAFB", "#00C853", "#A78BFA"]
        bars = ax5.bar(cat_counts.index, cat_counts.values, color=bar_colors, alpha=0.85, width=0.5)
        for bar in bars:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(int(bar.get_height())), ha='center', va='bottom',
                     color='white', fontsize=12, fontweight='bold')
        ax5.set_ylim(0, max(cat_counts.values) + 4)
        ax5.tick_params(colors='#c9d1d9', labelsize=10, rotation=15)
        for spine in ax5.spines.values():
            spine.set_color('rgba(255,255,255,0.08)')
        ax5.grid(axis='y', color='rgba(255,255,255,0.05)', linestyle='--')
        ax5.set_facecolor('#161b22')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

        # Stats table
        st.markdown("**Distribution Stats**")
        stats_df = pd.DataFrame({
            "Category": cat_counts.index,
            "Count": cat_counts.values,
            "Share %": [f"{v/len(df)*100:.1f}%" for v in cat_counts.values]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Word-length analysis
    st.markdown("---")
    st.markdown("**Resume Length Distribution by Category**")
    df["word_count"] = df["Resume_Details"].apply(lambda x: len(str(x).split()))
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    fig6.patch.set_facecolor('#0d1117')
    ax6.set_facecolor('#161b22')
    for i, (cat, color) in enumerate(zip(df["Category"].unique(), ["#FF6B35","#61DAFB","#00C853","#A78BFA"])):
        data = df[df["Category"] == cat]["word_count"]
        ax6.hist(data, bins=12, alpha=0.6, color=color, label=cat, edgecolor='#0d1117')
    ax6.legend(facecolor='#161b22', labelcolor='white', fontsize=10)
    ax6.set_xlabel("Word Count", color='#8b949e')
    ax6.set_ylabel("Frequency", color='#8b949e')
    ax6.tick_params(colors='#c9d1d9')
    for spine in ax6.spines.values():
        spine.set_color('rgba(255,255,255,0.08)')
    ax6.grid(color='rgba(255,255,255,0.05)', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📋 Dataset Viewer")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        cat_filter = st.selectbox("Filter by Category", ["All"] + list(df["Category"].unique()))
    with col_f2:
        search_term = st.text_input("Search keywords", placeholder="e.g. react, sql, workday...")

    display_df = df[["Category", "Resume_Details"]].copy()
    if cat_filter != "All":
        display_df = display_df[display_df["Category"] == cat_filter]
    if search_term:
        display_df = display_df[display_df["Resume_Details"].str.contains(search_term, case=False, na=False)]

    st.markdown(f"<div style='font-size:12px; color:#8b949e; margin-bottom:8px;'>Showing {len(display_df)} of {len(df)} records</div>", unsafe_allow_html=True)
    display_df["Preview"] = display_df["Resume_Details"].str[:200] + "..."
    st.dataframe(display_df[["Category", "Preview"]].reset_index(drop=True), use_container_width=True, height=420)

    # Download
    st.download_button(
        "⬇️ Download Full Dataset (CSV)",
        data=df.to_csv(index=False).encode(),
        file_name="Cleaned_Resumes.csv",
        mime="text/csv"
    )
