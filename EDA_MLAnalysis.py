# Streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Framingham Data Explorer",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Framingham Heart Study Data Explorer")
st.markdown("**Explore the dataset and feature correlations**")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a view:", ["Data Explorer", "Model Results"])

# Load the data
@st.cache_data 
def load_data():
    df = pd.read_csv('data/raw/framingham.csv')
    return df

# Load processed data
@st.cache_data
def load_processed_data():
    X_train = pd.read_csv('data/interim/X_train.csv')
    X_test  = pd.read_csv('data/interim/X_test.csv')
    y_train = pd.read_csv('data/interim/y_train.csv')
    y_test  = pd.read_csv('data/interim/y_test.csv')
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)
    return pd.concat([X, y], axis=1)

# Load data
df_original = load_data()
df_processed = load_processed_data()

# Sidebar - Data selection
data_choice = st.sidebar.selectbox("Choose dataset:", ["Original Data", "Processed Data"])

if data_choice == "Original Data":
    df = df_original
    st.sidebar.info("📝 Original data may contain missing values")
else:
    df = df_processed
    st.sidebar.success("✅ Processed data (missing values handled)")

# Display dataset info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Info")
st.sidebar.write(f"**Rows:** {df.shape[0]}")
st.sidebar.write(f"**Columns:** {df.shape[1]}")
st.sidebar.write(f"**Missing Values:** {df.isnull().sum().sum()}")

# Main content based on selection
if page == "Data Explorer":
    st.header("📊 Dataset Overview")

    # Display first few rows
    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    # Display statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    with col2:
        st.subheader("Data Types & Missing Values")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Missing': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)

    # Full dataset viewer
    if st.checkbox("Show full dataset"):
        st.subheader("Complete Dataset")
        st.dataframe(df, use_container_width=True)

    st.markdown("---")

if page == "Data Explorer":
    st.header("🔥 Correlation Matrix")
    
    # Calculate correlation
    correlation_matrix = df.corr()
    
    # Create two columns for different visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Heatmap Visualization")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            fmt='.2f', 
            square=True, 
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
    
    with col2:
        st.subheader("Correlation with Target")
        st.markdown("**TenYearCHD** (10-year CHD risk)")
        
        # Get correlations with target
        target_corr = correlation_matrix['TenYearCHD'].sort_values(ascending=False)
        
        # Create a dataframe for display
        corr_df = pd.DataFrame({
            'Feature': target_corr.index,
            'Correlation': target_corr.values
        })
        corr_df = corr_df[corr_df['Feature'] != 'TenYearCHD']  # Remove self-correlation
        
        # Color code by correlation strength
        def color_correlation(val):
            if val > 0.15:
                return 'background-color: #ffcccc'
            elif val > 0.05:
                return 'background-color: #ffe6cc'
            elif val < -0.05:
                return 'background-color: #ccf2ff'
            else:
                return ''
        
        styled_corr = corr_df.style.applymap(color_correlation, subset=['Correlation'])
        st.dataframe(styled_corr, use_container_width=True, height=500)
        
        st.markdown("""
        **Legend:**
        - 🔴 Strong positive (>0.15)
        - 🟠 Moderate positive (>0.05)
        - 🔵 Negative (<-0.05)
        - ⚪ Weak correlation
        """)
    
    # Interactive correlation explorer
    st.markdown("---")
    st.subheader("🔍 Correlation Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.selectbox("Select first feature:", df.columns)
    
    with col2:
        feature2 = st.selectbox("Select second feature:", df.columns)
    
    if feature1 and feature2:
        corr_value = correlation_matrix.loc[feature1, feature2]
        st.metric(
            label=f"Correlation: {feature1} vs {feature2}",
            value=f"{corr_value:.4f}"
        )
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[feature1], df[feature2], alpha=0.5, s=20)
        ax.set_xlabel(feature1, fontsize=12)
        ax.set_ylabel(feature2, fontsize=12)
        ax.set_title(f'{feature1} vs {feature2}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

if page == "Model Results":
    st.header("🤖 Model Results")
    st.markdown("Compare the four trained classifiers on the held-out test set.")

    # --- Helpers ---
    numerical_cols   = ['cigsPerDay', 'totChol', 'BMI', 'heartRate', 'glucose']
    categorical_cols = ['education', 'BPMeds']

    @st.cache_data
    def load_test_data():
        X_test = pd.read_csv('data/interim/X_test.csv')
        y_test = pd.read_csv('data/interim/y_test.csv').squeeze()
        X_train = pd.read_csv('data/interim/X_train.csv')
        return X_train, X_test, y_test

    @st.cache_resource
    def load_gb_model():
        return joblib.load('models/gradient_boosting.pkl')

    @st.cache_resource
    def load_rf_model():
        return joblib.load('models/random_forest.pkl')

    @st.cache_resource
    def load_svm_model():
        model  = joblib.load('models/svm.pkl')
        scaler = joblib.load('models/svm_scaler.pkl')
        return model, scaler

    @st.cache_resource
    def load_nn_model():
        import torch
        import torch.nn as nn

        class HeartDiseaseNetLogits(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
                    nn.Linear(64, 32),        nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
                    nn.Linear(32, 1)
                )
            def forward(self, x):
                return self.net(x)

        scaler = joblib.load('models/nn_scaler.pkl')
        model = HeartDiseaseNetLogits(input_dim=15)
        model.load_state_dict(torch.load('models/neural_network.pt', map_location='cpu'))
        model.eval()
        return model, scaler

    def impute(X_train, X_test):
        X_test = X_test.copy()
        num_imp = SimpleImputer(strategy='median')
        num_imp.fit(X_train[numerical_cols])
        X_test[numerical_cols] = num_imp.transform(X_test[numerical_cols])
        cat_imp = SimpleImputer(strategy='most_frequent')
        cat_imp.fit(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imp.transform(X_test[categorical_cols])
        return X_test

    def compute_metrics(y_test, y_pred, y_prob):
        report = classification_report(y_test, y_pred, output_dict=True)
        auc    = roc_auc_score(y_test, y_prob)
        cm     = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        return report, auc, cm, fpr, tpr

    X_train_raw, X_test_raw, y_test = load_test_data()
    X_test_imp = impute(X_train_raw, X_test_raw)

    # --- Model selector ---
    model_choice = st.selectbox(
        "Select model:", ["Gradient Boosting", "Random Forest", "Neural Network", "SVM"]
    )

    st.markdown("---")

    if model_choice == "Gradient Boosting":
        model = load_gb_model()
        y_pred  = model.predict(X_test_imp)
        y_prob  = model.predict_proba(X_test_imp)[:, 1]
        feat_img   = 'data/interim/feature_importance.png'
        roc_img    = 'data/interim/roc_curve.png'
        extra_label = "Feature Importance"
        extra_img   = feat_img

    elif model_choice == "Random Forest":
        model = load_rf_model()
        y_pred  = model.predict(X_test_imp)
        y_prob  = model.predict_proba(X_test_imp)[:, 1]
        feat_img   = 'data/interim/feature_importance_rf.png'
        roc_img    = 'data/interim/roc_curve_rf.png'
        extra_label = "Feature Importance"
        extra_img   = feat_img

    elif model_choice == "SVM":
        svm_model, svm_scaler = load_svm_model()
        X_scaled = svm_scaler.transform(X_test_imp)
        y_pred   = svm_model.predict(X_scaled)
        y_prob   = svm_model.predict_proba(X_scaled)[:, 1]
        roc_img    = 'data/interim/roc_curve_svm.png'
        extra_label = "Permutation Importance"
        extra_img   = 'data/interim/feature_importance_svm.png'

    else:  # Neural Network
        import torch
        nn_model, nn_scaler = load_nn_model()
        X_scaled = nn_scaler.transform(X_test_imp)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = nn_model(X_tensor).squeeze()
            y_prob = torch.sigmoid(logits).numpy()
        y_pred     = (y_prob >= 0.5).astype(int)
        roc_img    = 'data/interim/roc_curve_nn.png'
        extra_label = "Training Loss"
        extra_img   = 'data/interim/training_loss_nn.png'

    report, auc, cm, fpr, tpr = compute_metrics(y_test, y_pred, y_prob)

    # --- Key metrics ---
    st.subheader("Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",  f"{auc:.4f}")
    m2.metric("Accuracy",  f"{report['accuracy']:.4f}")
    m3.metric("Precision (CHD=1)", f"{report['1']['precision']:.4f}")
    m4.metric("Recall (CHD=1)",    f"{report['1']['recall']:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No CHD', 'CHD'], yticklabels=['No CHD', 'CHD']
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix — {model_choice}')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        # Classification report table
        st.subheader("Classification Report")
        rows = []
        for label in ['0', '1']:
            name = 'No CHD' if label == '0' else 'CHD'
            rows.append({
                'Class':     name,
                'Precision': f"{report[label]['precision']:.3f}",
                'Recall':    f"{report[label]['recall']:.3f}",
                'F1-Score':  f"{report[label]['f1-score']:.3f}",
                'Support':   int(report[label]['support']),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Macro / Weighted Averages")
        avg_rows = []
        for avg in ['macro avg', 'weighted avg']:
            avg_rows.append({
                'Average':   avg,
                'Precision': f"{report[avg]['precision']:.3f}",
                'Recall':    f"{report[avg]['recall']:.3f}",
                'F1-Score':  f"{report[avg]['f1-score']:.3f}",
            })
        st.dataframe(pd.DataFrame(avg_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("ROC Curve")
        st.image(roc_img, use_container_width=True)

    with col4:
        st.subheader(extra_label)
        st.image(extra_img, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>Framingham Heart Study Dataset</b></p>
    <p>Dataset for predicting 10-year risk of coronary heart disease (CHD)</p>
</div>
""", unsafe_allow_html=True)