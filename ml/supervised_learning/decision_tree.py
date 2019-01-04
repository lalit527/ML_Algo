import numpy as np

def divide_on_feature(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold
    
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])
    
    return np.array([X_1, X_2])

class DecisionNode:
    def __init__(self, feature_i=None, threshold=None, value=None, left_branch=None, right_branch=None):
        self.feature_i = feature_i        # Index for feature that is tested
        self.threshold = threshold        # Threshold value for a feature
        self.value = value                # Value if the node is a leaf in tree
        self.left_branch = left_branch    # Left Subtree
        self.right_branch = right_branch  # Right Subtree

        
class DecisionTree:
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf"), loss=None):
        self.root = None                              # Root node in dec. tree
        self.min_samples_split = min_samples_split    # The minimum number of samples required to split an internal node
        self.min_impurity = min_impurity              # The minimum impurity in a split  
        self.max_depth = max_depth                    # The maximum depth of the tree
        self.loss = loss                              # If Gradient Boosting
        self._impurity_calculation = None             # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None           # If y is one-hot encoded (multi-dim) or not (one-dim)
    
    def fit(self, X, y, loss=None):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = loss
    
    def _build_tree(self, X, y, current_depth=0):
        larget_impurity = 0
        best_criteria = None
        best_sets = None
        
        if self.one_dim:
            y = np.expand_dims(y, axis=1)
        
        Xy = np.concatenate((X, y), axis=1)
        
        n_samples, n_features = X.shape
        
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[: feature_i], axis=1)
                unique_values = np.unique(feature_values)
                
                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        
                        impurity = self._impurity_calculation(y, y1, y2)
                        
                        if impurity > larget_impurity:
                            larget_impurity = impurity
                            best_criteria = {"feature": feature_i, "threshold": threshold}
                            
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X of left subtree
                                "lefty": Xy1[:, n_features:],   # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]   # y of right subtree
                            }
                        
        if largest_impurity > self.min_impurity:
            left_branch = self._build_tree(best_sets.leftX, best_sets.lefty, current_depth + 1)
            right_branch = self._build_tree(best_sets.rightX, best_sets.righty, current_depth + 1)
            
            return DecisionNode(feature_i=best_criteria["feature"], threshold=best_criteria["threshold"], left_branch=left_branch, right_branch=right_branch)
        
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)
    
    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root
        
        if tree.value is not None:
            return tree.value
        
        feature_value = x[tree.feature_i]
        
        branch = tree.right_branch
        if isinstance(feature_value, int) or isinstance(threshold, float):
            if feature_value >= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch
    
        return self.predict_value(x, branch)
    
    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred 
    
    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        
        if tree.value is not None:
            print(tree.value)
            
        else:
            print ("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.left_branch, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.right_branch, indent + indent)
            
            
class DecisionTreeClassifier(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)
        return info_gain
    
    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common
    
    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super().fit(X, y)
        
class DecisionTreeRegressor(DecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        
        return sum(variance_reduction)
    
    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]
    
    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super().fit(X, y)
        
class XGBoostRegressionTree(DecisionTree):
    def _split(self, y):
        col = y.shape[1]//2
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred
    
    def _gain(self, y, y_pred):
        nominator = np.power((y * self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)
    
    def _gain_by_taylor(self, y, y1, y2):
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)
        
        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain
    
    def _approximate_update(self, y):
        y, y_pred = self._split(y)
        gradient = np.sum(y * self.loss.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation = gradient / hessian
        
        return update_approximation
    
    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super().fit(X, y)
        
def calculate_entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy