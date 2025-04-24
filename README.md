# 🚗 Used Car Price Predictor

## 📌 Overview
This project predicts the price of used cars in the US based on features like year, make, mileage, and condition. The model uses a structured machine learning pipeline to clean, preprocess, and train on data from Kaggle's "US Used Cars Dataset".

---

## 📦 Dataset
We used the [US Used Cars Dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset/data) from Kaggle.

To simplify the problem and reduce noise, we filtered the dataset to include only common US car makes and removed records with unrealistic or extreme outlier values.

---

## 🧹 Data Cleaning
- Removed columns with excessive null values
- Replaced smaller amounts of missing data with `"No Input"`
- Converted date and numeric object columns to proper float types
- Identified and removed outliers based on distribution plots
- Created a temporary `priceGroup` column to aid in test/train set splitting

---

## 🧪 Train/Test Split
To ensure reproducibility and prevent data leakage:
- Used a **hash-based** method on a unique identifier to split the dataset
- Verified that the test set's `priceGroup` proportions matched the full dataset
- Removed helper columns like `priceGroup` and `listing_id` before modeling

---

## 📊 Data Exploration & Visualization
- Conducted correlation analysis and visualized key relationships
- Used pair plots, histograms, and heatmaps to understand feature interactions and identify skewed distributions

---

## 🔧 Data Preprocessing
Built a full Scikit-Learn preprocessing pipeline:

- **Numerical pipeline**:  
  - Imputed missing values with median  
  - Scaled features using `StandardScaler()`

- **Boolean columns**:  
  - Converted to numeric using `OrdinalEncoder`

- **Categorical columns**:  
  - Transformed with `OneHotEncoder`

---

## 🤖 Model Training & Evaluation

### 1️⃣ Linear Regression
- Trained a baseline linear regression model
- Evaluated with cross-validation and standard regression metrics

### 2️⃣ Decision Tree Regressor
- Low training RMSE but high cross-validation RMSE
- Likely overfitting

### 3️⃣ Random Forest Regressor
- Better generalization performance
- Lower cross-validation RMSE than Decision Tree
- Chosen as the base for further tuning

---

## 🛠️ Hyperparameter Tuning
- Used `RandomizedSearchCV` to tune the Random Forest model
- Best hyperparameters:  
  `{'n_estimators': 400, 'min_samples_split': 5, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': False}`

### 🎯 Feature Importance
- Most important features (in order):  
  `year`, `horsepower`, `mileage`, ...

<p align="center">
  <img src="images/feature_importance.png" width="500"/>
</p>

---

## 🧪 Final Evaluation
- Evaluated the fine-tuned Random Forest on the test set
- Calculated RMSE and created a 95% confidence interval around the result
- Sample predictions with squared error analysis

---

## 💾 Saving the Model
- Final model saved using `joblib`
- Also saved:
  - The feature list
  - The transformation pipeline
  - A sample of the original training data

---

## 🧠 Key Takeaways
- Feature engineering and outlier removal significantly improved model reliability
- Cross-validation exposed overfitting in the Decision Tree model
- Random Forest with hyperparameter tuning provided the best generalization

---

## 🚀 Future Improvements
- Deploy the model with Streamlit for user interaction
- Add more external data (e.g., car condition scores, dealership data)
- Try deep learning approaches or ensemble stacking

---

## 📓 Full Notebook
You can view the full development process in this [Jupyter notebook](notebooks/Used_Car_Price_Predictor.ipynb).