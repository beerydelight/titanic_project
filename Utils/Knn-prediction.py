from Utils.Outlier_handler import outlier_handled
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

feature_columns = outlier_handled.columns.drop('survived')

X = outlier_handled[feature_columns]
y = outlier_handled['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict (returns 0 or 1)
y_pred = knn.predict(X_test)

# Evaluate
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title('KNN Classifier - Confusion Matrix')
plt.show()

# Get prediction probabilities (for class "survived")
y_proba = knn.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 5))
plt.hist(y_proba[y_test == 0], bins=20, alpha=0.7, label='Did Not Survive (Actual=0)', color='red')
plt.hist(y_proba[y_test == 1], bins=20, alpha=0.7, label='Survived (Actual=1)', color='green')
plt.axvline(0.5, color='k', linestyle='--', label='Decision Threshold')
plt.xlabel('Predicted Probability of Survival')
plt.ylabel('Frequency')
plt.legend()
plt.title('Prediction Probabilities by True Class')
plt.show()