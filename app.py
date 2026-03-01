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

# Always resolve paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(97,218,251,0.1);
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #61DAFB !important; }
.stApp { background: #0d1117; }
.main .block-container { padding: 2rem 2.5rem; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #8b949e !important; font-size: 11px !important;
    letter-spacing: 2px; text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #61DAFB !important; font-size: 26px !important; font-weight: 700 !important;
}
h1 { color: #ffffff !important; font-weight: 700 !important; }
h2, h3 { color: #c9d1d9 !important; }
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important; padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
[data-baseweb="tab"] {
    color: #8b949e !important; font-size: 13px !important;
    font-weight: 500 !important; letter-spacing: 1px !important;
}
[aria-selected="true"] {
    background: rgba(97,218,251,0.1) !important;
    color: #61DAFB !important; border-radius: 8px !important;
}
textarea {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important;
}
.stButton > button {
    background: linear-gradient(135deg,rgba(97,218,251,0.15),rgba(167,139,250,0.15)) !important;
    border: 1px solid rgba(97,218,251,0.4) !important;
    border-radius: 10px !important; color: #61DAFB !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
    padding: 0.6rem 1.5rem !important; width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg,rgba(97,218,251,0.25),rgba(167,139,250,0.25)) !important;
    border-color: #61DAFB !important;
}
.section-header {
    color: #8b949e; font-size: 11px; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
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

SAMPLES = {
    "🏢 PeopleSoft":   "Experienced PeopleSoft HCM Administrator with 5 years managing PIA environments. Skilled in Tuxedo, WebLogic, psadmin, PUM, Oracle database administration, and peopletools upgrades. Worked with HCM, FSCM modules in production support.",
    "⚛️ React JS Dev": "React JS Developer with expertise in Redux, hooks, context API, TypeScript, and Next.js. Built scalable frontend applications using component-based architecture, webpack, and modern JavaScript ES6+ for e-commerce platforms.",
    "🗄️ SQL Dev":      "SQL Developer with 4 years experience writing complex queries, stored procedures, and ETL pipelines. Proficient in MySQL, PostgreSQL, Oracle optimization, Tableau dashboards, and SSRS reporting for financial data.",
    "☁️ Workday":      "Workday HCM Consultant with Workday Studio integration experience. Implemented EIS integrations, BRE configuration, absence management, compensation, payroll, and recruitment modules for Fortune 500 clients."
}

PLOTLY_BASE = dict(
    paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
    font=dict(color='#c9d1d9', family='Space Grotesk'),
    margin=dict(l=10, r=10, t=45, b=10)
)

# ── Train all models (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
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
        "KNN":            KNeighborsClassifier(n_neighbors=3),
        "Decision Tree":  DecisionTreeClassifier(random_state=42),
        "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":            SVC(kernel='linear', probability=True, random_state=42),
        "Logistic Reg.":  LogisticRegression(max_iter=1000, random_state=42),
        "Bagging":        BaggingClassifier(random_state=42),
        "AdaBoost":       AdaBoostClassifier(random_state=42),
        "Gradient Boost": GradientBoostingClassifier(random_state=42),
        "Naive Bayes":    MultinomialNB(),
    }
    trained = {}
    for name, model in models.items():
        model.fit(x_tr, y_train)
        y_pred = model.predict(x_te)
        trained[name] = {
            "model":     model,
            "train_acc": round(model.score(x_tr, y_train) * 100, 1),
            "test_acc":  round(accuracy_score(y_test, y_pred) * 100, 1),
            "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 1),
            "recall":    round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 1),
            "f1":        round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 1),
            "y_pred":    y_pred,
            "report":    classification_report(y_test, y_pred, output_dict=True),
        }
    return tfidf, trained, df, y_test

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("🔄 Training models on dataset..."):
    tfidf, trained_models, df, y_test = train_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Resume Classifier")
    st.markdown("*ML-Powered Talent Intelligence*")
    st.markdown("---")
    st.markdown("### 🤖 Select Model")
    selected_model = st.selectbox("Model", list(trained_models.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown("""
    <div style='font-size:13px;line-height:2.2;color:#8b949e;'>
    📁 <b style='color:#c9d1d9;'>79</b> Resumes<br>
    🏷️ <b style='color:#c9d1d9;'>4</b> Categories<br>
    🔠 <b style='color:#c9d1d9;'>TF-IDF</b> Vectorizer<br>
    ✂️ <b style='color:#c9d1d9;'>75/25</b> Train / Test Split
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🏷️ Categories")
    for cat, meta in CATEGORY_META.items():
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:8px;margin:6px 0;padding:8px 10px;
             background:rgba(255,255,255,0.02);border-radius:8px;border-left:3px solid {meta["color"]};'>
            <span>{meta["icon"]}</span>
            <div>
                <div style='font-size:12px;color:#c9d1d9;font-weight:600;'>{cat}</div>
                <div style='font-size:10px;color:#8b949e;'>{meta["desc"]}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    acc = trained_models[selected_model]["test_acc"]
    st.markdown(f"""
    <div style='text-align:center;padding:16px;background:rgba(97,218,251,0.05);border-radius:10px;border:1px solid rgba(97,218,251,0.15);'>
        <div style='color:#61DAFB;font-size:28px;font-weight:700;'>{acc}%</div>
        <div style='color:#8b949e;font-size:10px;letter-spacing:2px;'>ACTIVE MODEL ACCURACY</div>
    </div>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin:0;font-size:2.2rem;'>
    📄 Resume <span style='color:#61DAFB;'>Classifier</span>
</h1>
<p style='color:#8b949e;margin:4px 0 24px;font-size:14px;letter-spacing:2px;'>
    ML-POWERED TALENT INTELLIGENCE SYSTEM
</p>""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Model",  selected_model)
m2.metric("Test Accuracy", f"{trained_models[selected_model]['test_acc']}%")
m3.metric("Total Resumes", "79")
m4.metric("Categories",    "4")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify Resume", "📊 Model Performance", "📈 Data Explorer", "📋 Dataset"])

# ════════════════════ TAB 1: CLASSIFY ════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-header">QUICK SAMPLES</div>', unsafe_allow_html=True)
        sc = st.columns(4)
        for i, (label, text) in enumerate(SAMPLES.items()):
            with sc[i]:
                if st.button(label, key=f"s{i}"):
                    st.session_state["txt"] = text

        st.markdown('<div class="section-header" style="margin-top:16px;">INPUT RESUME TEXT</div>', unsafe_allow_html=True)
        resume_text = st.text_area(
            "Resume", height=280,
            value=st.session_state.get("txt", ""),
            placeholder="Paste resume content here...",
            label_visibility="collapsed", key="txt"
        )
        words = len(resume_text.split()) if resume_text.strip() else 0
        st.markdown(f"<div style='font-size:11px;color:#484f58;text-align:right;'>{words} words · {len(resume_text)} chars</div>", unsafe_allow_html=True)
        classify_btn = st.button("⚡ Classify Resume", disabled=not resume_text.strip())

    with col_r:
        st.markdown('<div class="section-header">PREDICTION RESULT</div>', unsafe_allow_html=True)

        if not resume_text.strip():
            st.markdown("""
            <div style='height:340px;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;border:1px dashed rgba(255,255,255,0.08);
                        border-radius:14px;color:#484f58;text-align:center;gap:12px;'>
                <div style='font-size:48px;opacity:0.3;'>⚡</div>
                <div style='font-size:13px;'>Paste a resume and click Classify</div>
                <div style='font-size:11px;opacity:0.6;'>Or pick a Quick Sample on the left</div>
            </div>""", unsafe_allow_html=True)

        elif classify_btn or ("last_txt" in st.session_state and st.session_state.get("last_txt") == resume_text):
            st.session_state["last_txt"] = resume_text
            model_obj = trained_models[selected_model]["model"]
            vec = tfidf.transform([resume_text])
            pred = model_obj.predict(vec)[0]
            meta = CATEGORY_META[pred]

            if hasattr(model_obj, "predict_proba"):
                probs = model_obj.predict_proba(vec)[0]
                prob_dict = dict(zip(model_obj.classes_, probs))
                confidence = prob_dict[pred] * 100
            else:
                prob_dict = {pred: 1.0}
                confidence = 100.0

            st.markdown(f"""
            <div style='background:{meta["color"]}10;border:1px solid {meta["color"]}50;
                        border-radius:14px;padding:24px;text-align:center;margin-bottom:16px;
                        position:relative;overflow:hidden;'>
                <div style='position:absolute;top:0;left:0;right:0;height:3px;
                     background:linear-gradient(90deg,transparent,{meta["color"]},transparent);'></div>
                <div style='font-size:42px;margin-bottom:8px;'>{meta["icon"]}</div>
                <div style='font-size:11px;color:#8b949e;letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;'>PREDICTED CATEGORY</div>
                <div style='font-size:26px;font-weight:700;color:{meta["color"]};margin-bottom:4px;'>{pred}</div>
                <div style='font-size:12px;color:#8b949e;margin-bottom:20px;'>{meta["desc"]}</div>
                <div style='font-size:36px;font-weight:800;color:white;'>{confidence:.1f}%</div>
                <div style='font-size:11px;color:#8b949e;letter-spacing:2px;'>CONFIDENCE</div>
                <div style='margin-top:12px;background:rgba(0,0,0,0.3);border-radius:4px;height:8px;'>
                    <div style='width:{confidence}%;height:100%;background:{meta["color"]};
                         border-radius:4px;box-shadow:0 0 10px {meta["color"]}80;'></div>
                </div>
                <div style='margin-top:8px;font-size:11px;color:#484f58;'>
                    Model: <span style='color:#61DAFB;'>{selected_model}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            if hasattr(model_obj, "predict_proba"):
                sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                cats_  = [c for c, _ in sorted_probs]
                vals_  = [round(v * 100, 1) for _, v in sorted_probs]
                colors_= [CATEGORY_META[c]["color"] for c in cats_]
                fig = go.Figure(go.Bar(
                    x=vals_, y=cats_, orientation='h',
                    marker_color=colors_,
                    text=[f"{v}%" for v in vals_],
                    textposition='outside', textfont=dict(color='white', size=12)
                ))
                fig.update_layout(**PLOTLY_BASE, height=180,
                    title=dict(text="Category Probabilities", font=dict(size=13, color='#8b949e')),
                    xaxis=dict(range=[0, 118], showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False), showlegend=False,
                    margin=dict(l=10, r=60, t=35, b=10))
                st.plotly_chart(fig, use_container_width=True)

            found = [kw for kw in meta["keywords"] if kw in resume_text.lower()]
            if found:
                st.markdown('<div class="section-header" style="margin-top:4px;">MATCHING KEYWORDS</div>', unsafe_allow_html=True)
                chips = "".join([
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;'
                    f'margin:3px;background:{meta["color"]}15;border:1px solid {meta["color"]}30;'
                    f'color:{meta["color"]};font-family:monospace;">{kw}</span>'
                    for kw in found
                ])
                st.markdown(chips, unsafe_allow_html=True)

# ════════════════════ TAB 2: MODEL PERFORMANCE ═══════════════════════════════
with tab2:
    st.markdown("### 📊 Model Comparison")
    st.caption("All models trained on TF-IDF features · 75/25 stratified split")

    rows = []
    for name, data in trained_models.items():
        rows.append({
            "Model": name, "Train Acc %": data["train_acc"],
            "Test Acc %": data["test_acc"], "Precision %": data["precision"],
            "Recall %": data["recall"], "F1 Score %": data["f1"],
        })
    perf_df = pd.DataFrame(rows).sort_values("Test Acc %", ascending=False)

    def highlight(val):
        if isinstance(val, float):
            if val == 100.0: return "background:rgba(0,200,83,0.15);color:#00C853;font-weight:700;"
            elif val >= 90:  return "background:rgba(97,218,251,0.08);color:#61DAFB;"
            elif val < 60:   return "background:rgba(255,107,53,0.1);color:#FF6B35;"
        return "color:#c9d1d9;"

    styled = perf_df.style.applymap(
        highlight, subset=["Train Acc %","Test Acc %","Precision %","Recall %","F1 Score %"]
    ).format("{:.1f}", subset=["Train Acc %","Test Acc %","Precision %","Recall %","F1 Score %"])
    st.dataframe(styled, use_container_width=True, height=360)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        bar_colors = ["#00C853" if v==100 else "#61DAFB" if v>=90 else "#FF6B35" for v in perf_df["Test Acc %"]]
        fig1 = go.Figure(go.Bar(
            y=perf_df["Model"], x=perf_df["Test Acc %"], orientation='h',
            marker_color=bar_colors,
            text=[f"{v}%" for v in perf_df["Test Acc %"]],
            textposition='outside', textfont=dict(color='white')
        ))
        fig1.update_layout(**PLOTLY_BASE, height=380, title="Test Accuracy by Model",
            xaxis=dict(range=[0,118], title="Accuracy (%)", gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'), showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Test Acc', x=perf_df["Model"], y=perf_df["Test Acc %"],
                               marker_color='#61DAFB', opacity=0.85))
        fig2.add_trace(go.Bar(name='F1 Score', x=perf_df["Model"], y=perf_df["F1 Score %"],
                               marker_color='#A78BFA', opacity=0.85))
        fig2.update_layout(**PLOTLY_BASE, height=380, title="Test Accuracy vs F1 Score",
            barmode='group', xaxis=dict(tickangle=-35, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(bgcolor='rgba(0,0,0,0.3)', font=dict(color='white')))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"**Confusion Matrix — {selected_model}**")
    labels = sorted(df["Category"].unique())
    cm = confusion_matrix(y_test, trained_models[selected_model]["y_pred"], labels=labels)
    fig3 = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,'#0d1117'],[0.5,'#1e3a8a'],[1,'#3b82f6']],
        text=cm, texttemplate="%{text}", textfont=dict(size=18, color='white'),
        showscale=False
    ))
    fig3.update_layout(**PLOTLY_BASE, height=380,
        xaxis=dict(title="Predicted", side='bottom', gridcolor='rgba(0,0,0,0)'),
        yaxis=dict(title="Actual", autorange='reversed', gridcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"**Classification Report — {selected_model}**")
    report = trained_models[selected_model]["report"]
    report_rows = [
        {"Category": lbl,
         "Precision": f"{report[lbl]['precision']*100:.1f}%",
         "Recall":    f"{report[lbl]['recall']*100:.1f}%",
         "F1-Score":  f"{report[lbl]['f1-score']*100:.1f}%",
         "Support":   int(report[lbl]['support'])}
        for lbl in labels if lbl in report
    ]
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

# ════════════════════ TAB 3: DATA EXPLORER ═══════════════════════════════════
with tab3:
    st.markdown("### 📈 Dataset Explorer")
    cat_counts = df["Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    cat_colors = ["#FF6B35", "#61DAFB", "#00C853", "#A78BFA"]

    e1, e2 = st.columns(2)
    with e1:
        fig4 = go.Figure(go.Pie(
            labels=cat_counts["Category"], values=cat_counts["Count"],
            hole=0.55, marker=dict(colors=cat_colors, line=dict(color='#0d1117', width=3)),
            textinfo='label+percent', textfont=dict(color='white', size=12),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        ))
        fig4.update_layout(**PLOTLY_BASE, height=360, title="Category Distribution", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with e2:
        fig5 = go.Figure(go.Bar(
            x=cat_counts["Category"], y=cat_counts["Count"],
            marker_color=cat_colors,
            text=cat_counts["Count"], textposition='outside',
            textfont=dict(color='white', size=14),
            marker_line=dict(color='#0d1117', width=1)
        ))
        fig5.update_layout(**PLOTLY_BASE, height=360, title="Resume Count per Category",
            yaxis=dict(range=[0, cat_counts["Count"].max() + 5], gridcolor='rgba(255,255,255,0.05)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'), showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    df["word_count"] = df["Resume_Details"].apply(lambda x: len(str(x).split()))
    fig6 = go.Figure()
    for cat, color in zip(df["Category"].unique(), cat_colors):
        subset = df[df["Category"] == cat]["word_count"]
        fig6.add_trace(go.Histogram(x=subset, name=cat, marker_color=color, opacity=0.7, nbinsx=12))
    fig6.update_layout(**PLOTLY_BASE, height=360, barmode='overlay',
        title="Resume Word-Count Distribution by Category",
        xaxis=dict(title="Word Count", gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="Frequency", gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(bgcolor='rgba(0,0,0,0.4)', font=dict(color='white')))
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("**Word Count Stats by Category**")
    stats = df.groupby("Category")["word_count"].agg(["mean","min","max","std"]).round(0).astype(int)
    stats.columns = ["Mean Words", "Min Words", "Max Words", "Std Dev"]
    st.dataframe(stats, use_container_width=True)

# ════════════════════ TAB 4: DATASET ═════════════════════════════════════════
with tab4:
    st.markdown("### 📋 Dataset Viewer")
    c_f1, c_f2 = st.columns([2, 1])
    with c_f1:
        cat_filter = st.selectbox("Filter by Category", ["All"] + list(df["Category"].unique()))
    with c_f2:
        search_term = st.text_input("Search keywords", placeholder="e.g. react, workday...")

    disp = df[["Category","Resume_Details"]].copy()
    if cat_filter != "All":
        disp = disp[disp["Category"] == cat_filter]
    if search_term:
        disp = disp[disp["Resume_Details"].str.contains(search_term, case=False, na=False)]

    st.markdown(f"<div style='font-size:12px;color:#8b949e;margin-bottom:8px;'>Showing <b style='color:#61DAFB;'>{len(disp)}</b> of {len(df)} records</div>", unsafe_allow_html=True)
    disp["Preview"] = disp["Resume_Details"].str[:220] + "..."
    st.dataframe(disp[["Category","Preview"]].reset_index(drop=True), use_container_width=True, height=430)

    st.download_button(
        "⬇️ Download Full Dataset (CSV)",
        data=df.to_csv(index=False).encode(),
        file_name="Cleaned_Resumes.csv", mime="text/csv"
    )
