import streamlit as st
import pandas as pd
import joblib
import yaml
import os
import sys
import json
import plotly.express as px
import plotly.graph_objects as go

# Add the project root to the Python path to allow for module imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.model_evaluator import plot_confusion_matrix, plot_roc_curve
    
# --- Page Configuration ---
        st.set_page_config(
    page_title="Phishing Vulnerability Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
@st.cache_data
def load_data(path):
    """Loads data from a specified path with caching."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_json(path):
    """Loads a JSON file."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_resource
def load_model(path):
    """Loads a joblib model file."""
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_models(path):
    """Loads all joblib models from a directory."""
    models = {}
    if os.path.exists(path) and os.path.isdir(path):
        for file_name in os.listdir(path):
            if file_name.endswith('.joblib'):
                model_name = file_name.replace('.joblib', '').replace('_', ' ')
                model = joblib.load(os.path.join(path, file_name))
                models[model_name] = model
    return models

# --- Main App ---
def main():
    st.title("Phishing Vulnerability Analysis Dashboard")
    st.markdown("""
        This dashboard presents the results of the research project on human factors in phishing attacks.
        Navigate through the different sections using the sidebar to explore the data, statistical analysis,
        and machine learning model performance.
    """)

    # --- Load Data ---
    base_dir = project_root
    DATA_PATH = os.path.join(base_dir, 'data', 'processed', 'model_training_data.csv')
    STATS_REPORT_PATH = os.path.join(base_dir, 'results', 'statistical_analysis_report.json')
    CONFIG_PATH = os.path.join(base_dir, 'config', 'config.yaml')

    df = load_data(DATA_PATH)
    stats_report = load_json(STATS_REPORT_PATH)
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    if df is None:
        st.error("Processed data not found. Please run the preprocessing pipeline first (`python src/processing/preprocess_for_modeling.py`)")
        st.stop()
    
    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Statistical Insights", "ML Model Performance"])

    # --- Page Content ---
    if page == "Home":
        render_home()
    elif page == "Exploratory Data Analysis":
        render_eda(df)
    elif page == "Statistical Insights":
        render_stats(stats_report)
    elif page == "ML Model Performance":
        render_ml_performance(df, config)

def render_home():
    st.header("Project Overview")
    st.markdown("""
    This project aims to build a predictive model to identify individuals who are more vulnerable to phishing attacks based on psychological, behavioral, and demographic factors.
    
    **Key Goals:**
    - **Analyze Survey Data:** Understand the relationships between factors like stress, digital literacy, and cognitive biases.
    - **Predict Vulnerability:** Use machine learning to predict whether an individual will click on a simulated phishing link.
    - **Provide Insights:** Offer actionable insights for targeted cybersecurity training.
    
    Use the sidebar to explore the different analyses conducted.
    """)

def render_eda(df):
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("A first look at the distribution of key variables in our dataset.")
        
        col1, col2 = st.columns(2)
        
        with col1:
        st.subheader("Vulnerability Distribution")
        vuln_counts = df['vulnerable'].value_counts()
        fig_vuln = px.pie(values=vuln_counts.values, names=['Not Vulnerable', 'Vulnerable'], title='Proportion of Vulnerable Participants')
        st.plotly_chart(fig_vuln, use_container_width=True)
        
        with col2:
        st.subheader("Age Distribution")
        fig_age = px.histogram(df, x='D_AGE', title='Distribution of Participant Age Brackets', nbins=5)
        st.plotly_chart(fig_age, use_container_width=True)
    
    st.subheader("Correlation Heatmap of Scores")
    score_cols = [col for col in df.columns if '_score' in col]
    corr_matrix = df[score_cols].corr()
    fig_corr = go.Figure(data=go.Heatmap(
                   z=corr_matrix.values,
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   colorscale='Viridis'))
    fig_corr.update_layout(title='Correlation Matrix of Behavioral Scores')
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

def render_stats(report):
    st.header("Statistical Insights")

    if not report:
        st.warning("Statistical report not found. Please run the analysis script first (`python src/analysis/statistical_analysis.py`)")
        st.stop()
        
    for i in range(1, 5):
        rq_key = f"research_question_{i}"
        if rq_key in report:
            st.subheader(f"Research Question {i}: Analysis")
            data = report[rq_key]
            st.markdown(f"**Description:** {data['description']}")
            st.markdown(f"**Analysis Type:** `{data['analysis_type']}`")
            st.json(data)
            st.info(f"**Interpretation:** {data['interpretation']}")

def render_ml_performance(df, config):
    st.header("Machine Learning Model Performance")

    MODELS_PATH = os.path.join(project_root, 'models', 'saved')
    models = load_models(MODELS_PATH)
    
    if not models:
        st.error("No trained models found. Please run the training pipeline first (`python -m src.models.train_vulnerability_predictor`)")
        st.stop()

    model_names = list(models.keys())
    selected_model_name = st.selectbox("Select a model to evaluate:", model_names)
    
    model = models[selected_model_name]

    st.subheader(f"Performance for: {selected_model_name}")
    st.text(f"Model Type: {type(model).__name__}")
    
    # We don't have the test set here, so we will use a fresh split for demonstration
    # Note: This is for visualization only. The "real" metrics are in the MLflow run.
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    feature_cols_from_config = config['ml_models']['feature_columns']
    
    # --- EDGE CASE HANDLING ---
    # Filter config features to only those that actually exist in the loaded dataframe
    feature_cols = [col for col in feature_cols_from_config if col in df.columns]
    missing_cols = set(feature_cols_from_config) - set(feature_cols)
    if missing_cols:
        st.warning(f"The following feature columns from the config were not found in the data and will be ignored: {', '.join(missing_cols)}")

    target_col = config['ml_models']['target_variable']

    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the data. Cannot display model performance.")
        st.stop()

    # Re-prepare data for visualization
    df_vis = df.copy()
    categorical_cols = [col for col in df_vis.columns if df_vis[col].dtype == 'object' and col in feature_cols]
    for col in categorical_cols:
        le = LabelEncoder()
        df_vis[col] = le.fit_transform(df_vis[col])
        
    X = df_vis[feature_cols]
    y = df_vis[target_col]

    # --- EDGE CASE HANDLING ---
    # Check if there's more than one class to predict
    if y.nunique() < 2:
        st.warning("The dataset has only one outcome class (e.g., all 'vulnerable' or 'not vulnerable'). Model evaluation plots cannot be generated.")
        st.stop()

    # Handle NaNs that might exist
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True)
                
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
        
    st.subheader("Model Evaluation on a Sample Test Set")
        
        col1, col2 = st.columns(2)
        with col1:
        st.pyplot(plot_confusion_matrix(y_test, y_pred, model.classes_))
        with col2:
        st.pyplot(plot_roc_curve(y_test, y_pred_proba))

    # You could also load SHAP summary plot if it's saved as an artifact
    shap_img_path = os.path.join(project_root, 'outputs', 'figures', 'shap_summary.png')
    if os.path.exists(shap_img_path):
        st.subheader("Feature Importance (SHAP Summary)")
        st.image(shap_img_path)

if __name__ == "__main__":
    main()
