# logisticreg
Classification with Logistic Regression
# Breast Cancer Classification using Logistic Regression

This project builds a machine learning model to classify tumors as Benign (0) or Malignant (1) using the Breast Cancer Wisconsin dataset. The workflow includes EDA, preprocessing, logistic regression modeling, threshold tuning, and model evaluation.

---

## 1. Project Workflow

### Importing Libraries
Used Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn.

---

## 2. Loading the Dataset
The dataset contains the following important columns:
- id
- diagnosis (target)
- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- ... 
- fractal_dimension_worst
- Unnamed: 32 (removed later)

---

## 3. Data Cleaning and Preprocessing

Remove the unwanted column:
df = df.drop("Unnamed: 32", axis=1)

sql
Copy code

Convert diagnosis to numeric:
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

powershell
Copy code

Select features and target:
X = df.drop(['diagnosis', 'id'], axis=1)
y = df['diagnosis']

bash
Copy code

Train-test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

makefile
Copy code

Standardization:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

yaml
Copy code

---

## 4. Exploratory Data Analysis (EDA)
Performed:
- Checking missing values
- Summary statistics
- Distribution plots
- Correlation heatmap
- Boxplots to check outliers

---

## 5. Logistic Regression Model

Train the model:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

yaml
Copy code

Predict probabilities:
y_prob = model.predict_proba(X_test)[:, 1]

yaml
Copy code

---

## 6. Threshold Tuning

Default threshold is 0.5.  
Convert probabilities to class labels:

threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

yaml
Copy code

Threshold helps control the balance between precision and recall.

---

## 7. Model Evaluation

Confusion Matrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

mathematica
Copy code

Accuracy, Precision, Recall, F1-score:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

yaml
Copy code

---

## 8. Summary of Results
- Logistic Regression performed well for binary classification.
- Threshold tuning helps adjust recall and precision.
- Evaluation metrics give a complete understanding of model performance.

---

## 9. Requirements
Install required libraries:
pip install pandas numpy scikit-learn matplotlib seaborn

yaml
Copy code

---

## 10. Conclusion
This project demonstrates:
- How to clean and preprocess a dataset
- How logistic regression works for classification
- The importance of threshold
- How to use evaluation metrics such as confusion matrix, recall, and precision

Possible improvements:
- Try other models like SVM, Random Forest, XGBoost
- Add ROC-AUC curve
- Deploy using Streamlit
