# Import libraries
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel('INSERT Your AQI Data Here')

# Convert classification to numeric
df['Classification'] = df['Classification'].map({'Good': 0, 'Moderate': 1, 'Unhealthy': 2})

# Split data into train and test
X = df.drop('Classification', axis=1) 
y = df['Classification']

# Count class distribution before SMOTE
class_distribution_before = np.bincount(y)
print('Class Distribution Before SMOTE:', class_distribution_before)

# Reduce number of neighbors for SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ensure that the minimum class size is at least 6 for k_neighbors
min_class_size = min(np.bincount(y_train))
k_neighbors = min(5, min_class_size - 1)  # Adjust k_neighbors

oversample = SMOTE(k_neighbors=k_neighbors)

# Apply SMOTE
X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

# Count class distribution after SMOTE
class_distribution_after = np.bincount(y_train_resampled)
print('Class Distribution After SMOTE:', class_distribution_after)

# Combine before and after data for plotting
class_distribution_data = pd.DataFrame({
    'Class': ['Good', 'Moderate', 'Unhealthy', 'Good (SMOTE)', 'Moderate (SMOTE)', 'Unhealthy (SMOTE)'],
    'Count': np.concatenate([class_distribution_before, class_distribution_after])
})

# Plot class distribution before and after SMOTE
plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='Count', data=class_distribution_data, palette='viridis')
plt.title('Class Distribution Before and After SMOTE')
plt.ylabel('Number of Samples')
plt.show()



# Random Forest model  (trees=300, max_depth=None, min_samples_leaf=1, class_weight=None, gini, bootstrap=True, rs=42)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    class_weight=None,
    criterion='gini',
    bootstrap=True,
    random_state=42
)
rf.fit(X_train_resampled, y_train_resampled)

# Evaluation metrics
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')  
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')

# Feature Importance
feature_importance_rf = rf.feature_importances_
features_rf = X.columns

# Print metrics
print('Random Forest Evaluation Metrics:')
print('Accuracy:', acc_rf)
print('F1 Score:', f1_rf)
print('Precision:', precision_rf)
print('Recall:', recall_rf)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Cross-validation accuracy
cv_rf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=cv_rf, scoring='accuracy')
print('Cross-validation accuracy:', rf_scores.mean())

# Visualize all metrics for Random Forest
metrics_names_rf = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
metrics_values_rf = [acc_rf, f1_rf, precision_rf, recall_rf]

# Plotting metrics
plt.figure(figsize=(12, 6))
sns.barplot(x=metrics_values_rf, y=metrics_names_rf, palette='viridis')
plt.title('Random Forest Evaluation Metrics')
plt.xlabel('Metric Value')
plt.ylabel('Metrics')
plt.show()

# Plotting Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance_rf, y=features_rf, palette='viridis')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Balanced Random Forest model  (trees=300, max_depth=None, min_samples_leaf=1, class_weight='balanced', gini, bootstrap=True, rs=42)
brf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    class_weight='balanced',
    criterion='gini',
    bootstrap=True,
    random_state=42
) 
brf.fit(X_train_resampled, y_train_resampled)

# Evaluation metrics
y_pred_brf = brf.predict(X_test)
acc_brf = accuracy_score(y_test, y_pred_brf)
f1_brf = f1_score(y_test, y_pred_brf, average='weighted')  
precision_brf = precision_score(y_test, y_pred_brf, average='weighted')
recall_brf = recall_score(y_test, y_pred_brf, average='weighted')

# Feature Importance
feature_importance_brf = brf.feature_importances_
features_brf = X.columns

# Print metrics
print('Balanced Random Forest Evaluation Metrics:')
print('Accuracy:', acc_brf)
print('F1 Score:', f1_brf)
print('Precision:', precision_brf)
print('Recall:', recall_brf)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred_brf))
print(classification_report(y_test, y_pred_brf))

# Cross-validation accuracy
cv_brf = KFold(n_splits=5, shuffle=True, random_state=42)
brf_scores = cross_val_score(brf, X, y, cv=cv_brf, scoring='accuracy')
print('Cross-validation accuracy:', brf_scores.mean())

# Visualize all metrics for Balanced Random Forest
metrics_names_brf = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
metrics_values_brf = [acc_brf, f1_brf, precision_brf, recall_brf]

# Plotting metrics
plt.figure(figsize=(12, 6))
sns.barplot(x=metrics_values_brf, y=metrics_names_brf, palette='viridis')
plt.title('Balanced Random Forest Evaluation Metrics')
plt.xlabel('Metric Value')
plt.ylabel('Metrics')
plt.show()

# Plotting Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance_brf, y=features_brf, palette='viridis')
plt.title('Balanced Random Forest Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Shallow Random Forest model  (trees=100, max_depth=3, min_samples_leaf=1, gini, bootstrap=True, rs=42)
srf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    min_samples_leaf=1,
    class_weight=None,
    criterion='gini',
    bootstrap=True,
    random_state=42
)  
srf.fit(X_train_resampled, y_train_resampled)

# Evaluation metrics
y_pred_srf = srf.predict(X_test)
acc_srf = accuracy_score(y_test, y_pred_srf)
f1_srf = f1_score(y_test, y_pred_srf, average='weighted')  
precision_srf = precision_score(y_test, y_pred_srf, average='weighted')
recall_srf = recall_score(y_test, y_pred_srf, average='weighted')

# Feature Importance
feature_importance_srf = srf.feature_importances_
features_srf = X.columns

# Print metrics
print('Shallow Random Forest Evaluation Metrics:')
print('Accuracy:', acc_srf)
print('F1 Score:', f1_srf)
print('Precision:', precision_srf)
print('Recall:', recall_srf)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred_srf))
print(classification_report(y_test, y_pred_srf))

# Cross-validation accuracy
cv_srf = KFold(n_splits=5, shuffle=True, random_state=42)
srf_scores = cross_val_score(srf, X, y, cv=cv_srf, scoring='accuracy')
print('Cross-validation accuracy:', srf_scores.mean())

# Visualize all metrics for Shallow Random Forest
metrics_names_srf = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
metrics_values_srf = [acc_srf, f1_srf, precision_srf, recall_srf]

# Plotting metrics
plt.figure(figsize=(12, 6))
sns.barplot(x=metrics_values_srf, y=metrics_names_srf, palette='viridis')
plt.title('Shallow Random Forest Evaluation Metrics')
plt.xlabel('Metric Value')
plt.ylabel('Metrics')
plt.show()

# Plotting Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance_srf, y=features_srf, palette='viridis')
plt.title('Shallow Random Forest Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Extra Trees model  (trees=200, max_depth=None, min_samples_leaf=1, gini, bootstrap=True, rs=42)
erf = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    criterion='gini',
    bootstrap=True,        # explicit per table (sklearn default is False)
    random_state=42
)
erf.fit(X_train_resampled, y_train_resampled)

# Evaluation metrics
y_pred_erf = erf.predict(X_test)
acc_erf = accuracy_score(y_test, y_pred_erf)
f1_erf = f1_score(y_test, y_pred_erf, average='weighted')  
precision_erf = precision_score(y_test, y_pred_erf, average='weighted')
recall_erf = recall_score(y_test, y_pred_erf, average='weighted')

# Feature Importance
feature_importance_erf = erf.feature_importances_
features_erf = X.columns

# Print metrics
print('Extra Trees Evaluation Metrics:')
print('Accuracy:', acc_erf)
print('F1 Score:', f1_erf)
print('Precision:', precision_erf)
print('Recall:', recall_erf)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred_erf))
print(classification_report(y_test, y_pred_erf))

# Cross-validation accuracy
cv_erf = KFold(n_splits=5, shuffle=True, random_state=42)
erf_scores = cross_val_score(erf, X, y, cv=cv_erf, scoring='accuracy')
print('Cross-validation accuracy:', erf_scores.mean())

# Visualize all metrics for Extra Trees
metrics_names_erf = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
metrics_values_erf = [acc_erf, f1_erf, precision_erf, recall_erf]

# Plotting metrics
plt.figure(figsize=(12, 6))
sns.barplot(x=metrics_values_erf, y=metrics_names_erf, palette='viridis')
plt.title('Extra Trees Evaluation Metrics')
plt.xlabel('Metric Value')
plt.ylabel('Metrics')
plt.show()

# Plotting Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance_erf, y=features_erf, palette='viridis')
plt.title('Extra Trees Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Identify the best model based on accuracy
best_model_name = max({'Random Forest': acc_rf, 'Balanced Random Forest': acc_brf, 'Shallow Random Forest': acc_srf, 'Extra Trees': acc_erf}.items(), key=lambda x: x[1])[0]

# Visualize the performance of the best model
best_model_values = {'Random Forest': metrics_values_rf, 'Balanced Random Forest': metrics_values_brf, 'Shallow Random Forest': metrics_values_srf, 'Extra Trees': metrics_values_erf}[best_model_name]

plt.figure(figsize=(12, 6))
sns.barplot(x=best_model_values, y=metrics_names_rf, palette='viridis')
plt.title(f'Best Model: {best_model_name} Evaluation Metrics')
plt.xlabel('Metric Value')
plt.ylabel('Metrics')
plt.show()
best_model_name = max({'Random Forest': acc_rf, 'Balanced Random Forest': acc_brf, 'Shallow Random Forest': acc_srf, 'Extra Trees': acc_erf}.items(), key=lambda x: x[1])[0]
print(best_model_name)

