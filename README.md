# Energy_Efficiency_Project

## 📘 Overview
This project focuses on analyzing and predicting the **Energy Efficiency of Buildings** using various **Machine Learning algorithms**.  
The aim is to study how different structural and environmental parameters affect the **Heating Load (Y1)** and **Cooling Load (Y2)** of buildings and to develop predictive models that improve building energy performance.

---

## 🎯 Problem Statement
The main goal is to evaluate how different building design parameters impact energy efficiency and to build models that can predict heating and cooling loads accurately.  
By doing so, we can contribute to **sustainable energy use**, **cost reduction**, and **environment-friendly designs**.

---

## 🧠 Aim
To implement and compare different machine learning algorithms for analyzing and predicting the **energy efficiency** of buildings based on input features such as wall area, roof area, glazing area, and orientation.

---

## 📊 Dataset Description
- **Dataset Name:** Energy Efficiency Dataset  
- **Source:** UCI Machine Learning Repository  
- **Total Samples:** 768  
- **Attributes:**
  | Parameter | Description |
  |------------|-------------|
  | X1 | Relative Compactness |
  | X2 | Surface Area |
  | X3 | Wall Area |
  | X4 | Roof Area |
  | X5 | Overall Height |
  | X6 | Orientation |
  | X7 | Glazing Area |
  | X8 | Glazing Area Distribution |
  | Y1 | Heating Load |
  | Y2 | Cooling Load |

---

## 🔬 Experiments Performed

### **1️⃣ Experiment 1: Data Preprocessing & Visualization**
- Handled missing values and checked dataset consistency.  
- Visualized data using:
  - **Histogram:** To see the distribution of heating and cooling loads.  
  - **Boxplot:** To detect outliers in load values.  
  - **Pie Chart:** To show proportion of different features affecting loads.  
  - **Correlation Heatmap:** To identify relationships between features.

---

### **2️⃣ Experiment 2: K-Means Clustering**
- Performed clustering to group buildings with similar energy profiles.  
- Evaluated with **Elbow Method** to find optimal number of clusters.  
- Helped understand natural groupings in data.

---

### **3️⃣ Experiment 3: PCA & SVD (Dimensionality Reduction)**
- **PCA (Principal Component Analysis):**  
  Reduced dataset dimensions while keeping maximum variance.  
  Used **standard scaling** before PCA to normalize all features.  
- **SVD (Singular Value Decomposition):**  
  Decomposed the matrix to analyze latent relationships and reduce redundancy.

---

### **4️⃣ Experiment 4: Support Vector Machine (SVM)**
- Classified **Energy Efficiency Levels (High/Low)** based on Heating Load.  
- Used **Linear and Kernel SVM** models.  
- Evaluated using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
- Visualized **Decision Boundaries**, **Support Vectors**, and **Margins**.

---

### **5️⃣ Experiment 5: Ensemble Learning**
- Implemented:
  - **Random Forest Classifier**
  - **Gradient Boosting Classifier**
- Compared their performance using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - Confusion Matrices  
- Both models showed strong predictive performance.

---

### **6️⃣ Experiment 6: Polynomial / Nonlinear Regression**
- Modeled **nonlinear relationships** between input features and energy loads.  
- Visualized results using a **3D Surface Plot** to show predicted vs actual values.

---

## ⚙️ Technologies Used
- **Programming Language:** Python  
- **Libraries:**
  - pandas, numpy, matplotlib, seaborn  
  - scikit-learn  
  - plotly (for visualization)  

---

## 📈 Key Insights
- **Relative Compactness**, **Surface Area**, and **Overall Height** are the most influential features.  
- PCA helped simplify the dataset without major information loss.  
- SVM and Ensemble Models achieved high accuracy (>95%) in predicting efficiency levels.  
- Polynomial Regression successfully modeled complex nonlinear patterns.

---

## ✅ Conclusion
The project successfully demonstrated how **machine learning** can be applied to **predict and analyze energy efficiency** in buildings.  
Using ensemble models and SVM, we achieved high performance and meaningful insights that can aid in designing **energy-optimized structures**.


---

