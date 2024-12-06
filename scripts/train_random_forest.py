from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

# Preprocess the data
data = preprocess_data('data/student-mat.csv')

# Split into features and target
X = data.drop('G3', axis=1)
y = (data['G3'] >= 10).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
with open('results/model_performance_random_forest.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))
