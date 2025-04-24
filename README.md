# 🚗 Used Car Price Predictor

## 📌 Project Overview
This project demonstrates a full end-to-end **data science pipeline** to predict the price of used cars in the US market. Using **Python** and powerful tools like **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **Scikit-Learn**, I cleaned, explored, and modeled real-world data to produce accurate price predictions.

It showcases hands-on experience with **machine learning**, **feature engineering**, **hyperparameter tuning**, and **model evaluation**, making it an ideal demonstration for **data science job applications**.

---

## 📦 Dataset
- **Source**: [US Used Cars Dataset from Kaggle](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset/data)
- The dataset contains listings for used cars including fields like make, model, year, mileage, condition, and price.
- I filtered the dataset to only include **common US car brands** and removed extreme outliers to ensure data quality.

---

## 🧹 Data Cleaning & Preparation
Performed thorough **data preprocessing** using **Python** and **Pandas**:
- Dropped columns with excessive missing values
- Replaced light missing data with placeholders ("No Input")
- Converted data types (e.g., dates, numerical strings) into proper numeric formats
- Removed **outliers** using visualizations and domain logic
- Created a temporary `priceGroup` column for stratified splitting

---

## 🧪 Train/Test Splitting
To ensure **reproducibility** and prevent **data leakage**, I used a **hash-based method**:
- Each row was assigned to train or test based on a hash of its unique ID
- This technique guarantees that the test set remains consistent over time
- Removed unnecessary columns (`priceGroup`, `listing_id`) before modeling

---

## 📊 Data Exploration & Visualization
Used **Seaborn** and **Matplotlib** for **exploratory data analysis (EDA)**:
- Investigated feature distributions and relationships
- Analyzed correlations between features and target (`price`)
- Visualized skewness and feature importance drivers

---

## 🔧 Preprocessing Pipelines (Scikit-Learn)

Built a full **Scikit-Learn transformation pipeline**:

### 🔢 Numerical Pipeline:
- Handled missing values using `SimpleImputer(strategy='median')`
- Scaled features using `StandardScaler()`

### ⚙️ Categorical and Boolean Encoding:
- Boolean columns (`is_new`, `franchise_dealer`) encoded using `OrdinalEncoder()`
- Object columns transformed via `OneHotEncoder()`

---

## 🤖 Machine Learning Models

Implemented and compared multiple **regression models** using **Scikit-Learn**:

### 1️⃣ Linear Regression
- Used as a baseline model
- Evaluated with cross-validation and regression metrics

### 2️⃣ Decision Tree Regressor
- Showed strong fit on training set
- **Overfitting** revealed through high cross-validation RMSE

### 3️⃣ Random Forest Regressor
- Outperformed Decision Tree in cross-validation
- Chosen for **robust generalization** and better model stability

---

## 🛠️ Hyperparameter Tuning

Used **RandomizedSearchCV** to tune the **Random Forest**:
- Best parameters:  
  `{'n_estimators': 400, 'min_samples_split': 5, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': False}`
- Visualized and ranked **feature importances** based on model output

<p align="center">
  <img src="images/feature_importance.png" width="500"/>
</p>

---

## 📈 Final Evaluation

Evaluated the final, tuned **Random Forest model** on the **test set**:
- Calculated **Root Mean Squared Error (RMSE)** and other regression metrics
- Computed a **95% confidence interval** around RMSE
- Sampled squared prediction errors to assess model variance

---

## 💾 Model Deployment

Saved the final model pipeline using **Joblib**:
- Serialized the trained model
- Saved the preprocessing pipeline and feature list
- Included a sample of the original training data for future validation

---

## 🧠 Key Takeaways

- Demonstrated ability to clean, preprocess, and engineer real-world data
- Built **machine learning models** with **Scikit-Learn**
- Applied **cross-validation** and **hyperparameter tuning**
- Emphasized **interpretability** through feature importance
- Practiced full **data science lifecycle**, from raw data to deployable model

---

## 🚀 Future Enhancements

- Deploy as a **web app** using **Streamlit** for interactive price prediction
- Expand with external sources (e.g., car condition scores, location data)
- Explore **deep learning** or **ensemble stacking** methods for improved performance

---

## 📓 Full Project Notebook

👉 [Click here to view the full Jupyter Notebook](notebooks/Used_Car_Price_Predictor.ipynb)

---

## 🛠️ Tech Stack
- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**
- **Scikit-Learn**
- **Joblib**
- **Jupyter Notebook**
