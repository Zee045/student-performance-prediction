from preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Preprocess data
data = preprocess_data('../data/student-mat.csv')

# Split data
X = data.drop('G3', axis=1)
y = (data['G3'] >= 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Save results
with open('../results/model_performance_logistic.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred_lr))
