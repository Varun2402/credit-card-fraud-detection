
# Credit Card Fraud Detection using Machine Learning

This project is an **end-to-end implementation** of a **Credit Card Fraud Detection system** built for my IDS C2P2 Project. It leverages advanced techniques like **XGBoost**, **SMOTE (for imbalanced data handling)**, and **feature scaling** to accurately identify fraudulent transactions from real-world data.

## Key Highlights

* Trained on a real-world dataset with **284,807 transactions**
* Handled **extreme class imbalance** using **SMOTE (Synthetic Minority Oversampling Technique)**
* Applied **XGBoost** classifier for high-performance fraud prediction
* Achieved **ROC AUC Score: 0.98+** with precision and recall optimized for fraud cases
* Includes **visualizations**, **confusion matrix**, and **classification report**
* Structured for **scalability** and easy **deployment**

## Dataset Description
The dataset contains 284,807 transactions, among which only 492 are frauds (Class = 1).
### Features:
* **Time:** Seconds elapsed between the transaction and the first transaction in the dataset.
* **Amount:** Transaction amount.
* **Class:** Target variable (0 = Not Fraud, 1 = Fraud).
Remaining features (V1 to V28) are principal components from PCA to ensure privacy.

