import numpy as np
import pandas as pd
from scipy.optimize import minimize

class ErrorLDA:
    outcomes = None
    outcome_fractions = {}
    means = {}
    variance = None

    num_calls_debug = 0

    def __init__(self):
        pass

    # For a single fixed mean (assuming a constant class)
    def negative_log_loss_fixed_mean(self, X_train, Sigma, X_minus_mu, X_train_errors=None):
        # TODO: Surely at least one of these could be done more efficiently
        # TODO: Handle the "no training errors" case
        X_overall_errors = X_train_errors.apply(lambda x: x + Sigma)

        first_term = X_overall_errors.apply(lambda E: np.log(np.linalg.det(E))).sum()

        #second_term = sum((row @ np.linalg.inv(X_overall_errors[i]) @ row) for i, row in X_minus_mu.iterrows())

        # SUBSTANTIALLY more efficient than it was initially
        second_term = np.einsum('ij,ijk,ik->i',
                                X_minus_mu.values,
                                np.stack(X_overall_errors.apply(np.linalg.inv).values),
                                X_minus_mu.values).sum()

        return 0.5*(first_term + second_term)
    

    def negative_log_loss(self, X_train_splits, Sigma, X_minus_mu_splits, X_train_errors_splits=None):

        total_log_loss = 0.0

        for c in self.outcomes:
            current_log_loss = self.negative_log_loss_fixed_mean(X_train_splits[c], Sigma, X_minus_mu_splits[c],
                                                                 X_train_errors=X_train_errors_splits[c])
            
            total_log_loss += current_log_loss
            
        return total_log_loss


    def fit(self, X_train, y_train, X_train_errors=None):
        self.outcomes = list(y_train.unique())
        self.outcomes.sort()

        X_train_splits = {}
        X_minus_mu_splits = {}
        X_train_errors_splits = {}

        # TODO: Perhaps far more efficient to only split the data once
        #       and not split every single time.
        for c in self.outcomes:
            X_train_outcome = X_train[y_train == c]

            self.outcome_fractions[c] = len(X_train_outcome.index) / len(X_train.index)

            # Self-explanatory
            self.means[c] = X_train_outcome.mean(axis=0)
            
            # Save for when we try to optimize, just to avoid having to re-split over and over again 
            X_train_splits[c] = X_train_outcome           
            X_minus_mu_splits[c] = X_train_outcome.apply(lambda x: x.values - self.means[c], axis=1)
            X_train_errors_splits[c] = X_train_errors[y_train == c]

            

        # Perform gradient descent
        # First, we parameterize Sigma using Log-Cholesky parametrization
        n = len(X_train.columns) # Sigma is n by n
        n_lower = (n-1)*n // 2 # elements of the strictly lower triangular matrix
        n_diag = n # number of elements on the diagonal (obvious)

        params = np.zeros(n_lower + n_diag)

        def get_L(params):
            L = np.zeros(shape=(n,n))

            # The strictly lower triangular part
            row = 0
            col = 0
            for i in range(0, n_lower):
                if row == col:
                    col = 0
                    row += 1
                L[row,col] = params[i]
                col += 1

            # The diagonal part
            for i in range(0, n_diag):
                L[i,i] = np.exp(params[n_lower + i])

            return L
        
        def get_Sigma(params):
            L = get_L(params)
            return np.matmul(L, L.T)
        
        """
        learning_rate = 1.0
        gradient_calc_step = 0.1
        epsilon = 0.000001
        improvement = 100.0 # Dummy value for initial "improvement" in loss

        Sigma = get_Sigma(params) # The default value of Sigma, should be the identity matrix
        loss = self.negative_log_loss(X_train, y_train, Sigma, X_train_errors=X_train_errors)

        while improvement > epsilon:
            gradient = np.zeros(params.shape[0])

            for i in range(0,n):
                test_params = params.copy()
                test_params[i] += gradient_calc_step
                test_Sigma = get_Sigma(test_params)
                test_loss = self.negative_log_loss(X_train, y_train, test_Sigma, X_train_errors=X_train_errors)

                gradient[i] = (test_loss - loss) / gradient_calc_step

            print(gradient)

            params -= gradient * learning_rate
            Sigma = get_Sigma(params)
            old_loss = loss
            loss = self.negative_log_loss(X_train, y_train, Sigma, X_train_errors=X_train_errors)

            improvement = old_loss - loss
            print(loss)

            # NOTE: This could actually end one step too late, on an increased value of loss
        """

        self.num_calls_debug = 0

        def objective_function(params):
            self.num_calls_debug += 1

            # Diagonal entries are generated as e^(...)
            # Let's avoid overflow and other problems.
            for i in range(n_lower, n_lower + n_diag):
                if params[i] > 20.0:
                    return 1000.0 * 1000.0 * 1000.0

            return self.negative_log_loss(X_train_splits, get_Sigma(params), X_minus_mu_splits, X_train_errors_splits=X_train_errors_splits)
        
        best_params = minimize(objective_function, params.copy()).x
        print("Objective function called {0} times".format(self.num_calls_debug))
        print(best_params)
        print(get_Sigma(best_params))

        self.variance = get_Sigma(best_params)



    # Kind of assuming for now that X_train and y_train are pandas arrays
    # I might convert this to numpy code later.
    def fit_old(self, X_train, y_train, X_train_errors=None):
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
            
            #if X_train_errors is not None:
            #    errors_outcome = X_train_errors[y_train == c]
            #    for error in errors_outcome:
            #        self.variance = self.variance - error

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

            if exponents[key] >= 10.0:
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
        print("Means")
        for key in self.means:
            print(self.means[key])

        print()
        print("Outcomes (classes)")
        print(self.outcomes)

        print()
        print("Outcome fractions")
        for key in self.outcome_fractions:
            print(self.outcome_fractions[key])

        print()
        print("The underlying variance matrix")
        print(self.variance)

        print()
        print("Eigenvalues of the underlying variance matrix")
        print(np.linalg.eig(self.variance)[0])