import numpy as np
import pandas as pd

class ErrorLDA:
    outcomes = None
    outcome_fractions = {}
    means = {}
    variance = None

    def __init__(self):
        pass

    # Kind of assuming for now that X_train and y_train are pandas arrays
    # I might convert this to numpy code later.
    def fit(self, X_train, y_train, X_train_errors=None):
        self.outcomes = list(y_train.unique())
        self.outcomes.sort()

        # Build the variance by adding to it and then rescaling
        self.variance = np.zeros(shape=(len(X_train.columns), len(X_train.columns)))
        
        for c in self.outcomes:
            X_train_outcome = X_train[y_train == c]

            self.outcome_fractions[c] = len(X_train_outcome.index) / len(X_train.index)

            # Self-explanatory
            self.means[c] = X_train_outcome.mean(axis=0)

            # TODO: Actually compute the proper MLE estimator
            # This is more of an approximation.
            self.variance += X_train_outcome.cov().values * float(len(X_train_outcome.index))
            
            if X_train_errors is not None:
                errors_outcome = X_train_errors[y_train == c]
                for error in errors_outcome:
                    self.variance = self.variance - error

        self.variance /= len(X_train.index)

    def predict_proba_one(self, x, x_error=None):

        exponents = {}

        for c in self.outcomes:
            # Scalar, fraction in that class
            mu_c = self.means[c]
            Sigma = self.variance
            D = np.zeros(shape=(len(x.columns), len(x.columns))) if x_error is None else x_error

            # Internal computations
            h = np.linalg.inv(D).dot(x) + np.linalg.inv(Sigma).dot(mu_c)
            B = np.linalg.inv(np.linalg.inv(D) + np.linalg.inv(Sigma))

            # P(x|c) is proportional to this
            exponent = (1/2) * (B.dot(h).dot(h) - np.linalg.inv(Sigma).dot(mu_c).dot(mu_c))

            exponents[c] = exponent 

        # Numerical stability stuff.
        # Things are proportional to e^(...), but those exponents can be HUGE.
        # Hence, we will subtract off the minimum one.

        min_exponent = min(exponents.values())
        total_proportion = 0.0
        proportions = []

        still_too_large = None

        for key in exponents:
            exponents[key] -= min_exponent

            if exponents[key] >= 10:
                still_too_large = key
                break

            total_proportion += self.outcome_fractions[key] * np.exp(exponents[key])

        # Something would still have way too large of an exponent
        # Kind of janky and assumes something doesn't have comparable size
        # Not good for classifying >= 3 things
        #
        # TODO: FIX PROPERLY!
        #
        if still_too_large is not None:
            for key in exponents:
                proportions.append(1.0 if key == still_too_large else 0.0)
            return proportions

        # Nothing too large
        for key in exponents:
            proportions.append(self.outcome_fractions[key] * np.exp(exponents[key]) / total_proportion)

        return proportions
    
    def predict_proba(self, X, X_error=None):
        predictions = []

        for i in range(0, len(X.index)):
            predictions.append(self.predict_proba_one(X.iloc[i],
                                                      x_error=None if X_error is None else X_error.iloc[i]))
            
        return pd.DataFrame(predictions, columns=self.outcomes)

    
    def debug_func(self):
        for key in self.means:
            print(self.means[key])

        print(self.outcomes)

        for key in self.outcome_fractions:
            print(self.outcome_fractions[key])

        print(self.variance)