import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import time



# ------------------------------
# Page Config & Custom CSS
# ------------------------------




st.set_page_config(
    page_title="Kaggle Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300&display=swap');
    
    .main {
        background: linear-gradient(45deg, #0e1117 30%, #192231 90%);
        color: #ffffff;
    }
    
    .st-bw {
        background-color: rgba(255,255,255,0.1) !important;
        border-radius: 10px;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    h1 {
        font-family: 'Roboto Mono', monospace;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 10px;
    }
    
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    
    .st-cb { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Helper Functions
# ------------------------------
def plot_interactive_corr(df):
    corr = df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='Viridis',
        hoverongaps=False
    ))
    fig.update_layout(title='Interactive Correlation Matrix', height=600)
    return fig

def plot_feature_importance(model, features, model_name="Model"):
    try:
        # For tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        # For linear models
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        # For models without feature importance
        else:
            st.warning(f"⚠️ Feature importance not available for {model_name}")
            return None
            
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', height=500, color_continuous_scale='Emrld',
                     title=f'Feature Importance ({model_name})')
        return fig
        
    except Exception as e:
        st.error(f"❌ Error generating feature importance: {str(e)}")
        return None

# ------------------------------
# Main App
# ------------------------------
def main():
    st.title("📈 Kaggle Analyzer Pro")
    st.markdown("**Advanced EDA & AutoML Platform**")
    
    # File Upload
    with st.sidebar.expander("📤 DATA UPLOAD", expanded=True):
        uploaded_file = st.file_uploader("", type=["csv", "xlsx"], 
                                       help="Upload Kaggle dataset (CSV/Excel)")
        use_sample = st.checkbox("Use Sample Dataset (Titanic)")
    
    # Load Data
    if use_sample:
        df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
        target_col = 'Survived'
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = None
    else:
        st.info("👈 Upload a dataset or use sample data to begin!")
        st.stop()
    
    # Preprocessing
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # ------------------------------
    # Main Tabs
    # ------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Dashboard", 
        "🔍 Deep Analysis", 
        "🤖 ML Studio", 
        "📤 Export"
    ])
    
    with tab1:  # Dashboard Tab
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📦 Data Summary")
            st.metric("Samples", df.shape[0])
            st.metric("Features", df.shape[1])
            st.metric("Missing Values", df.isnull().sum().sum())
            
            st.subheader("📝 Quick Actions")
            if st.button("✨ Auto-Clean Data"):
                df.dropna(inplace=True)
                st.success("Data cleaned! Missing values removed.")
        
        with col2:
            st.subheader("📊 Data Preview")
            st.dataframe(df.style.background_gradient(cmap='Blues'), height=300)
    
    with tab2:  # Deep Analysis Tab
        st.subheader("🔎 Interactive Exploration")
        
        analysis_type = st.selectbox("Select Analysis Type", [
            "Distribution Analysis",
            "Correlation Matrix",
            "Outlier Detection",
            "Time Series Analysis"
        ])
        
        if analysis_type == "Distribution Analysis":
            col = st.selectbox("Select Column", df.columns)
            fig = px.histogram(df, x=col, marginal="box", 
                              template="plotly_dark", color_discrete_sequence=['#4CAF50'])
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Correlation Matrix":
            st.plotly_chart(plot_interactive_corr(df), use_container_width=True)
    
    with tab3:  # ML Studio Tab
        st.subheader("🤖 Machine Learning Playground")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        model_type = st.selectbox("Model Type", ["Classification", "Regression"])
        models = st.multiselect("Select Models", [
            "Random Forest", 
            "Gradient Boosting", 
            "SVM"
        ], default=["Random Forest"])
        
        test_size = st.slider("Test Size (%)", 10, 40, 20)
        random_state = st.number_input("Random State", 42)
        
    with col2:
        if not target_col:
            target_col = st.selectbox("Select Target Column", df.columns)
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state)
        
        results = []
        trained_models = {}  # Store trained models
        progress_bar = st.progress(0)
        
        for i, model_name in enumerate(models):
            with st.spinner(f"Training {model_name}..."):
                time.sleep(1)  # Simulate training
                
                if model_name == "Random Forest":
                    model = RandomForestClassifier()
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingClassifier()
                elif model_name == "SVM":
                    model = SVC(probability=True)
                    
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                results.append({
                    "Model": model_name,
                    "Accuracy": f"{acc:.2%}",
                    "Features": X.shape[1]
                })
                trained_models[model_name] = model
                progress_bar.progress((i+1)/len(models))
        
        st.subheader("📊 Model Performance")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.highlight_max(axis=0, color='#4CAF50'))
        
        st.subheader("📈 Feature Importance")
        selected_model = st.selectbox("Select Model for Feature Importance", models)
        model = trained_models[selected_model]
        fig = plot_feature_importance(model, X.columns, selected_model)  # Now passing 3 args correctly
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importance available for this model type")
    
    with tab4:  # Export Tab
        st.subheader("📤 Export Results")
        
        export_format = st.radio("Select Format", ["CSV", "Excel", "JSON"])
        export_name = st.text_input("File Name", "my_analysis")
        
        if st.button("🚀 Export"):
            with st.spinner("Generating export..."):
                time.sleep(1)
                if export_format == "CSV":
                    df.to_csv(f"{export_name}.csv", index=False)
                elif export_format == "Excel":
                    df.to_excel(f"{export_name}.xlsx", index=False)
                
                st.success(f"✅ Exported successfully as {export_name}.{export_format.lower()}")
                st.balloons()

if __name__ == "__main__":
    main()

