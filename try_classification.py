import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt
from dataset_generation import DatasetGenerator, TOPVIEWRODENTS_CONFIG, W_SIZE

dataset = DatasetGenerator(TOPVIEWRODENTS_CONFIG)
dataset.generate_stat_dataset(W_SIZE, 0, 'boosting_dataset.npy', 0.8)

dataset = np.load('boosting_dataset.npy')

X = dataset[:, :-1]
print(X.shape)
y = dataset[:, -1]
for i in range(len(TOPVIEWRODENTS_CONFIG['classes'])):
    print(f"Number of samples of class {i} is {X[y == i].shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create and train the CatBoostClassifier
model = CatBoostClassifier(
    iterations=300, depth=2, learning_rate=0.1,
   loss_function='MultiClass', verbose=True, task_type='GPU', devices='0')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
ap = average_precision_score(y_test, model.predict_proba(X_test))
print(f"CatBoostClassifier mAP: {ap}")

joblib.dump(model, 'catboost_model.pkl')
