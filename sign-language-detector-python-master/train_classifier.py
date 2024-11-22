import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# List to store accuracy values for different number of trees
accuracy_values = []

# Vary the number of trees from 1 to 100 and train the model for each value
for n_trees in range(1, 101):
    model = RandomForestClassifier(n_estimators=n_trees)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    accuracy_values.append(accuracy)

# Plotting the accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), accuracy_values, marker='o', linestyle='-')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Trees')
plt.grid(True)
plt.show()

# Choose the best number of trees based on the accuracy graph
best_n_trees = np.argmax(accuracy_values) + 1
print('Best number of trees:', best_n_trees)

# Train the final model with the best number of trees
final_model = RandomForestClassifier(n_estimators=best_n_trees)
final_model.fit(x_train, y_train)

# Evaluate the final model
y_predict = final_model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')
conf_matrix = confusion_matrix(y_test, y_predict)

print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1-score: {:.2f}'.format(f1))
print('Confusion Matrix:')
print(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the final model
with open('model.p', 'wb') as f:
    pickle.dump({'model': final_model}, f)