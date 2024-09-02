from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    # TODO: return the distance between the two input data points.
    ed = np.sqrt(np.sum((point1 - point2)**2))
    return ed
    

def knn_classifier(train_data, train_labels, test_data, k=3):
    """Simple k-NN classifier."""
    predictions = []

    for test_point in test_data:
        # Calculate distances from the current test point to all training points
        distances = [euclidean_distance(test_point, train_point) for train_point in train_data]

        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # TODO: Get the labels of the k nearest neighbors
        k_nearest_labels = [train_labels[i] for i in k_indices]

        # TODO: Determine the most common label among the k nearest neighbors
        most_common_label = Counter(k_nearest_labels).most_common()[0][0]


        # Append the predicted label to the predictions list
        predictions.append(most_common_label)

    return np.array(predictions)

df = pd.read_csv(r'C:/Users/adwan/Documents/!AIML231/data/banknotes_new.csv')
X, y = df.drop('Class', axis=1),  df['Class']

# TODO: Use function StandardScaler().fit_transform() to normalize the value range of each feature in the dataset.
X = StandardScaler().fit_transform(X)

# TODO: Conduct training data and test data split randomly with a 50:50 ratio, setting random_state=100
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5,random_state = 100)
y_train = y_train.values
y_test = y_test.values

# Set the value of k.
k=3
test_label = knn_classifier(X_train, y_train, X_test, k=k)
print(test_label)

# Calculate and print the accuracy of the knn classifier on the test dataset.
accuracy = 1.0-np.sum(np.abs(y_test - test_label))/y_test.shape[0]
print(f'The accuracy of the knn classifier is {accuracy} when k={k}')
