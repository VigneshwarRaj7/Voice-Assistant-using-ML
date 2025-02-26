import numpy as np
from scipy.optimize import minimize

class MAELinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.
        :param X: Training data features, shape (n_samples, n_features)
        :param y: Training data targets, shape (n_samples,)
        """
        # Add intercept term to X
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        # Define the loss function for minimization
        def mae_loss(theta, X, y):
            return np.mean(np.abs(X @ theta - y))
        # Initial guess for the parameters
        initial_theta = np.zeros(X_b.shape[1])
        # Minimize the MAE loss
        result = minimize(mae_loss, initial_theta, args=(X_b, y))
        self.theta = result.x

    def predict(self, X):
        """
        Make predictions using the linear model.
        :param X: Samples for which to predict, shape (n_samples, n_features)
        :return: Predicted values, shape (n_samples,)
        """
        if self.theta is None:
            raise ValueError("Model must be fitted before making predictions.")
        # Add intercept term to X
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b @ self.theta

# Example usage
if __name__ == "__main__":
    # Example data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    # Creating and fitting the model
    model = MAELinearRegression()
    model.fit(X, y)
    
    # Making predictions
    X_new = np.array([[6], [7]])  # New samples for prediction
    predictions = model.predict(X_new)
    print("Predictions:", predictions)
