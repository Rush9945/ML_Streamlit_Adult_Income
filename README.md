
# Adult Income Classification (Census Income)

## a) Problem statement
The goal is to build a machine learning model that predicts whether an indivusual's income is
">50K" or "<=50K" based on demographic & employements realated attributes. & this is binary 
classification problem.

## b) Dataset description
- The dataset used is the Adult Census Income dataset from UCI ML repo.
- This dataset has 32,562 rows &  14 attributes & 1 target. Mixed categorical & numeric.
- arget attribute is `income` (`<=50K`, `>50K`).

## c) Models used & comparison table
Six models were trained on the same dataset. Reported metrics: Accuracy, AUC, Precision, Recall, F1, MCC.

| ML Model Name              | Accuracy |  AUC   | Precision | Recall |   F1   |  MCC   |
|--------------------------- |----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.8543   | 0.9136 |  0.7502   | 0.6218 | 0.6800 | 0.5911 |
| Decision Tree              | 0.8152   | 0.7510 |  0.6303   | 0.6232 | 0.6267 | 0.5039 |
| kNN                        | 0.8341   | 0.8672 |  0.6832   | 0.6218 | 0.6511 | 0.5436 |
| Naive Bayes                | 0.6010   | 0.8300 |  0.3795   | 0.9487 | 0.5421 | 0.3876 |
| Random Forest (Ensemble)   | 0.8598   | 0.9114 |  0.7579   | 0.6418 | 0.6950 | 0.6083 |
| XGBoost (Ensemble)         | 0.8767   | 0.9342 |  0.7933   | 0.6824 | 0.7337 | 0.6572 |

## Observations about model performance
Based on review the results.

| ML Model Name              | Observation about model performance |
|---------------------------|--------------------------------------|
| Logistic Regression        | Good balanced performance with strong AUC & stable generalization|
| Decision Tree              | Moderate accuracy; tends to overfit compared to ensemble models  |
| kNN                        | Decent overll results but slightly weak recall,sensitive to distance based patterns |
| Naive Bayes                | Very high recall but low precision, leading to many false positives | 
| Random Forest (Ensemble)   | Strong & consistent|performance acress all metrics with good balance |
| XGBoost (Ensemble)         | Best overall performer with highest accuracy, AUC & MCC |
