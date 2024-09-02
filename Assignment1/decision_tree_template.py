"""
This is a Python code template for building a simple decision tree classifier.

The decision tree classifier follows a binary tree structure, meaning that each internal node has two separate branches, i.e., the left child node and the right child node.
Each internal node checks a condition regarding a specific value of a categorial feature (we assume that all features of the classification problem are categorical features).
If the condition is true (the feature has the specific categorial value indicated in the node), we move to the left child node to continue the classification process.
On the other hand, if the condition is false (the feature has a different value from the value specified), we move to the right child node to continue the classification process.

The information gain metric is used as the splitting method at each internal node of the decision tree classifier. In other words, we choose a condition "feature==value" that
will lead to the highest information gain as the condition whenever we split an internal node.

To control the complexity of the decision tree classifier, the code template currently uses the maximum tree depth as a control condition. Whenever a node of the decision
tree reaches the maximum tree depth provided by the user, no further splitting will be performed. The corresponding node will be treated as a leaf node. You can add
an additional condition based on the minimum partition size requirement for complexity control.
"""

import pandas as pd
import numpy as np

class DecisionTreeNode:
    def __init__(self, feature=None, value=None, left=None, right=None, is_leaf=False, label=None, depth=0):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.label = label
        self.depth = depth

    def string_representation(self):
        if self.is_leaf:
            return ''.join(self.depth*['  '],) + f'leaf:{self.feature}={self.value}&label={self.label}&depth={self.depth}'
        else:
            return ''.join(self.depth*['  '],) + f'{self.feature}={self.value}&depth={self.depth}\n->left  ' + self.left.string_representation() + '\n->right ' + self.right.string_representation()


class DecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None


    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)


    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1 or len(y) < 2: # TODO: add additional condition based on the mininum partition size requirement for complexity control.
            return DecisionTreeNode(is_leaf=True, label=max(set(y), key=list(y).count), depth=depth)

        best_feature, best_value = self._find_best_split(X, y)
        left_idx = X[best_feature] == best_value
        right_idx = ~left_idx

        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return DecisionTreeNode(feature=best_feature, value=best_value, left=left_child, right=right_child, depth=depth)


    def _find_best_split(self, X, y):
        best_feature, best_value, best_gain = None, None, 0
        for feature in X.columns:
            values = list(set(X[feature]))
            values.sort()
            for value in values:
                left_idx = X[feature] == value
                right_idx = ~left_idx
                gain = self._information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = value
        return best_feature, best_value


    def _information_gain(self, parent, left, right): # TODO: implement this function to calculate the information gain.
        def entropy(y):
            counts = np.bincount(y)
            probabilities = counts / len(y)
            return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
            
        parent_entropy = entropy(parent)
        left_entropy = entropy(left)
        right_entropy = entropy(right)
        
        total_samples = len(parent)
        weighted_avg_entropy = (len(left) / total_samples) * left_entropy + (len(right) / total_samples) * right_entropy
        # TODO: implement the remaining part of this function to calculate the information gain.
        information_gain = parent_entropy - weighted_avg_entropy

        return information_gain


    def predict(self, X):
        tmp_pred = []
        for index, row in X.iterrows():
            tmp_pred.append(self._predict_sample(self.root, row))
        return tmp_pred


    def _predict_sample(self, node, sample):
        # TODO: implement this function to make predictions based on a node and its child nodes of a decision tree.
        if node.is_leaf:
            return node.label

        if sample[node.feature] == node.value:
            return self._predict_sample(node.left, sample)
        else:
            return self._predict_sample(node.right, sample)



### Code to test the learning of decision trees.
data = pd.DataFrame({
    'Outlook': ['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r'],
    'Humidity': ['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h'],
    'Wind': ['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'],
    'PlayTennis': [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
})

# Train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(data[['Outlook', 'Humidity', 'Wind']], data['PlayTennis'])

# Make predictions
predictions = clf.predict(pd.DataFrame({
    'Outlook': ['o', 's'],
    'Humidity': ['n', 'h'],
    'Wind': ['w', 's']
}))

print(predictions)
print(clf.root.string_representation())
