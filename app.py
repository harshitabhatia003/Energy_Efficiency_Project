import streamlit as st
import os
import joblib
from PIL import Image

# ===============================================
# ⚡ ENERGY EFFICIENCY PROJECT DASHBOARD
# ===============================================

st.set_page_config(page_title="Energy Efficiency Dashboard", layout="wide")
st.title("⚡ Energy Efficiency ML Dashboard")
st.markdown("### Select an Experiment to view saved visualizations and models")

# Folder paths
MODEL_DIR = os.path.join(os.getcwd(), "model")
IMAGE_DIR = os.path.join(os.getcwd(), "images")

# ===============================================
# 🔹 Helper Functions
# ===============================================
def load_model(file_name):
    """Load model safely"""
    model_path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success(f"✅ Loaded model: {file_name}")
        return model
    else:
        st.error(f"❌ Model '{file_name}' not found in model folder.")
        return None

def show_image(file_name, caption=None):
    """Display image safely"""
    img_path = os.path.join(IMAGE_DIR, file_name)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.error(f"❌ Image '{file_name}' not found in images folder.")

# ===============================================
# 🔹 Sidebar Experiment Selector
# ===============================================
exp = st.sidebar.selectbox(
    "🧪 Choose Experiment",
    (
        "Experiment 1: Data Visualization",
        "Experiment 2: Linear Regression",
        "Experiment 3: Decision Tree",
        "Experiment 4: SVM",
        "Experiment 5: Ensemble (Random Forest & Gradient Boosting)",
        "Experiment 6: Polynomial Regression",
        "Experiment 7: K-Means Clustering",
        "Experiment 8: PCA & SVD"
    )
)

# ===============================================
# 🔸 Experiment 1: Data Visualization
# ===============================================
if "Visualization" in exp:
    st.header("📊 Experiment 1: Data Visualization")
    show_image("exp1_histogram.png", "Histogram for Feature Distributions")
    show_image("exp1_boxplot.png", "Boxplot to Detect Outliers")
    show_image("exp1_piechart.png", "Pie Chart - Mean Contribution of Features")
    show_image("exp1_heatmap.png", "Correlation Heatmap")

# ===============================================
# 🔸 Experiment 2: Linear Regression
# ===============================================
elif "Linear" in exp:
    st.header("📈 Experiment 2: Linear Regression")
    load_model("linear_regression.pkl")
    show_image("exp2_linear_regression.png", "Linear Regression Fit")

# ===============================================
# 🔸 Experiment 3: Decision Tree
# ===============================================
elif "Decision" in exp:
    st.header("🌳 Experiment 3: Decision Tree")
    load_model("decision_tree.pkl")
    show_image("exp3_decision_tree.png", "Decision Tree Visualization")

# ===============================================
# 🔸 Experiment 4: SVM
# ===============================================
elif "SVM" in exp:
    st.header("⚙️ Experiment 4: Support Vector Machine (SVM)")
    load_model("svm_model.pkl")
    show_image("exp4_svm_boundary.png", "SVM Decision Boundary")
    show_image("exp4_svm_confusion_matrix.png", "SVM Confusion Matrix")

# ===============================================
# 🔸 Experiment 5: Ensemble Learning
# ===============================================
elif "Ensemble" in exp:
    st.header("🌲 Experiment 5: Ensemble Learning")
    load_model("random_forest.pkl")
    show_image("exp5_random_forest.png", "Random Forest Visualization")
    show_image("exp5_gradient_boosting.png", "Gradient Boosting Visualization")

# ===============================================
# 🔸 Experiment 6: Polynomial Regression
# ===============================================
elif "Polynomial" in exp:
    st.header("🧮 Experiment 6: Polynomial Regression (3D Surface)")
    load_model("polynomial_regression.pkl")
    show_image("exp6_polynomial_regression_3d.png", "3D Polynomial Regression Plot")

# ===============================================
# 🔸 Experiment 7: K-Means Clustering
# ===============================================
elif "K-Means" in exp:
    st.header("🎯 Experiment 7: K-Means Clustering & Elbow Method")
    show_image("exp7_elbow_method.png", "Elbow Method Graph")
    show_image("exp7_kmeans_clusters.png", "K-Means Cluster Visualization")

# ===============================================
# 🔸 Experiment 8: PCA & SVD
# ===============================================
elif "PCA" in exp:
    st.header("🔻 Experiment 8: PCA & SVD (Dimensionality Reduction)")
    load_model("pca_svd_model.pkl")
    show_image("pca_explained_variance.png", "PCA Explained Variance")
    show_image("pca_scatter_plot.png", "PCA 2D Scatter Plot")
    show_image("svd_scatter_plot.png", "SVD 2D Scatter Plot")

# ===============================================
# 🏁 Footer
# ===============================================
st.markdown("---")
st.markdown("👩‍💻 **Developed by Harshita Bhatia and Hiten Bhurani** | ML Experiment Dashboard powered by Streamlit")
