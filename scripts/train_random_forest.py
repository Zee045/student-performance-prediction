from preprocess import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Preprocess data
data = preprocess_data('data/student-mat.csv')

# Split data
X = data.drop('G3', axis=1)
y = (data['G3'] >= 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Save results
with open('results/model_performance_random_forest.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred_rf))
