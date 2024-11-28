import numpy as np
import pandas as pd
from scipy.optimize import minimize

class ErrorLDA:
    outcomes = None
    outcome_fractions = None
    means = {}
    variance = None
    error_scaler = None

    num_calls_debug = 0

    def __init__(self):
        pass

    # For a single fixed mean (assuming a constant class)
    def negative_log_loss_fixed_mean(self, X_train, Sigma, X_minus_mu, X_train_errors=None, error_scaler=None):

        # Convert scale errors appropriately
        X_overall_errors = X_train_errors + Sigma if error_scaler is None else (X_train_errors @ error_scaler) + Sigma

        first_term = np.log(np.linalg.det(X_overall_errors)).sum()

        # SUBSTANTIALLY more efficient than it was initially
        second_term = np.einsum('ij,ijk,ik->i',
                                X_minus_mu,
                                np.linalg.inv(X_overall_errors),
                                X_minus_mu).sum()

        return 0.5*(first_term + second_term)
    

    def negative_log_loss(self, X_train_splits, Sigma, X_minus_mu_splits,
                          X_train_errors_splits=None, error_scaler=None):

        total_log_loss = 0.0

        for c in self.outcomes:
            current_log_loss = self.negative_log_loss_fixed_mean(X_train_splits[c], Sigma, X_minus_mu_splits[c],
                                                                 X_train_errors=X_train_errors_splits[c], error_scaler=error_scaler)
            
            total_log_loss += current_log_loss
            
        return total_log_loss


    def fit(self, X_train, y_train, X_train_errors=None, error_scaling=False):
        # For now, use pandas for convenience.
        # However, in the actual optimization function, we will use pure numpy
        # TODO: Might as well do this in pure numpy at some point
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        if not isinstance(X_train_errors, pd.Series):
            X_train_errors = pd.Series(X_train)

        # Get rid of annoying index nonsense
        # Note that this will also create copies of the data, to avoid messing around with the original values.
        # TODO: CHECK THESE ARE COPIES!
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_train_errors = X_train_errors.reset_index(drop=True)

        self.outcomes = list(y_train.unique())
        self.outcomes.sort()

        self.outcome_fractions = np.zeros(len(self.outcomes))

        X_train_splits = {}
        X_minus_mu_splits = {}
        X_train_errors_splits = {}

        # TODO: Perhaps far more efficient to only split the data once
        #       and not split every single time.
        for i_c, c in enumerate(self.outcomes):
            X_train_c = X_train[y_train == c]

            self.outcome_fractions[i_c] = len(X_train_c.index) / len(X_train.index)

            # Self-explanatory
            self.means[c] = X_train_c.mean(axis=0)
            
            # Save for when we try to optimize, just to avoid having to re-split over and over again.
            # Note that these will be saved as pure numpy arrays, rather than dataframes.
            # Speed is of the utmost importance here.
            X_train_splits[c] = X_train_c.to_numpy()
            X_minus_mu_splits[c] = X_train_c.apply(lambda row: row.to_numpy() - self.means[c], axis=1).to_numpy()
            X_train_errors_splits[c] = np.stack(X_train_errors[y_train == c].to_numpy())

        # First, we parameterize Sigma using Log-Cholesky parametrization
        # That is, Sigma = LL^T, where L = M + e^D, where M is strictly lower diagonal (zeros on diagonal), and D is diagonal.

        n = len(X_train.columns) # Sigma is n by n
        n_lower = (n-1)*n // 2 # elements of the strictly lower triangular matrix
        n_diag = n # number of elements on the diagonal (obvious)

        # Strictly lower entries of L (i.e. M above)
        # Parameterized diagonal entries of L (i.e. D, which gets turned into e^D)
        # Final few parameters are reserved for variance-scaling, if that gets used.
        params = np.zeros(n_lower + n_diag + n_diag)

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
        
        def get_scaler(params):
            #return np.diag(1 / (1 + np.exp( - params[n_lower + n_diag : ] ))) if error_scaling else np.identity(n)
            return np.diag(params[n_lower + n_diag : ]) + np.identity(n) if error_scaling else np.identity(n)
        
        # Just to keep track of how often the objective function was called. For testing performance.
        self.num_calls_debug = 0

        def objective_function(params):
            self.num_calls_debug += 1

            # Diagonal entries are generated as e^(...)
            # Let's avoid overflow and other problems.
            for i in range(n_lower, n_lower + n_diag):
                if params[i] > 20.0:
                    return 1000.0 * 1000.0 * 1000.0 # Huge penalty

            return self.negative_log_loss(X_train_splits, get_Sigma(params), X_minus_mu_splits,
                                          X_train_errors_splits=X_train_errors_splits, error_scaler=get_scaler(params))
        
        # A rough initial estimate. The original covariance matrix, minus the average error.
        Sigma_guess = X_train.cov().to_numpy()
        if X_train_errors is not None:
            Sigma_guess -= X_train_errors.sum() / len(X_train_errors.index)
        # Deal with small, potentially negative eigenvalues in the above matrix
        min_eigenvalue = np.min(np.linalg.eig(Sigma_guess)[0])
        if min_eigenvalue < 1.0:
            Sigma_guess -= (min_eigenvalue - 1) * np.identity(n) # Minimum eigenvalue of 1 now
        
        def Sigma_to_params(Sigma):
            L = np.linalg.cholesky(Sigma)
            L = L @ np.diag(np.sign(np.diag(L))) # Correct for non-positive stuff on the diagonal
            
            result = np.zeros(n_lower + n_diag)

            # The strictly lower triangular part
            row = 0
            col = 0
            for i in range(0, n_lower):
                if row == col:
                    col = 0
                    row += 1
                result[i] = L[row,col]
                col += 1

            # The diagonal part
            result[n_lower : ] = np.log(np.diag(L))

            return result

        # Initial guess for Sigma as above, plus identity matrix for the scaler.
        x0 = np.concatenate([Sigma_to_params(Sigma_guess), np.zeros(n_diag)]) if error_scaling else Sigma_to_params(Sigma_guess)

        # Recall that the scaler is centered at the identity matrix. Bounds represent deviations on each diagonal entry there.
        bounds = [(None, None)] * (n_lower + n_diag) + [(-0.9, 0.1)] * n_diag if error_scaling else [(None, None)] * (n_lower + n_diag)

        # There are technically no bounds present when we have no error scaling. Hopefully this forces things to be sped up.
        best_params = minimize(objective_function, x0=x0, bounds=bounds).x if error_scaling else minimize(objective_function, x0=x0).x

        self.variance = get_Sigma(best_params)
        self.error_scaler = get_scaler(best_params)

    
    def predict_proba(self, X, X_error=None):
        # Convert to nummpy, for efficiency purposes
        # TODO: Check data types!
        X = X.to_numpy()
        X_error = np.stack(X_error.to_numpy())

        # Rescaled if appropriate
        X_error = X_error if self.error_scaler is None else X_error @ self.error_scaler

        # Store results. Columns correspond to the individual classes.
        # Probabilities are proportional to e^(...), but that exponent can be huge.
        # For numerical stability reasons, we will fix that before computing probabilities.
        exponents = np.zeros(shape=(X.shape[0], len(self.outcomes)))
        probabilities = np.zeros(shape=(X.shape[0], len(self.outcomes)))

        # For efficiency, can be computed ahead of time.
        Sigma_inv = np.linalg.inv(self.variance)

        for i_c, c in enumerate(self.outcomes):
            mu_c = self.means[c]

            # Some internal computations that pop out of the math.
            # h = D^{-1}x + \Sigma^{-1}\mu
            # B = {D^{-1} + \Sigma^{-1}}^{-1}
            h = np.einsum('ijk,ik->ij', np.linalg.inv(X_error), X) + (Sigma_inv @ mu_c)
            B = np.linalg.inv(np.linalg.inv(X_error) + Sigma_inv)

            # Things are proportional to (class probability) * e^((1/2) * (<Bh,h> + <\Sigma^{-1}\mu,\mu>))
            exponents[:,i_c] = (1/2) * (np.einsum('ij,ijk,ik->i', h, B, h) - (mu_c @ Sigma_inv @ mu_c))

        # Compute the minimum exponent among classes, and subtract it off.
        # This avoids things like comparing e^4000 and e^4002.
        min_exponents = np.min(exponents, axis=1)
        exponents -= min_exponents.reshape((-1,1))

        # Test for exponents that are still to large.
        # TODO: This is a little janky, and will not really give proper answers
        #       if multiple exponents are extremely large. (only possible if >= 3 classes)
        #       This SHOULD probably fix this eventually.
        large_exponents = np.any(exponents >= 20.0, axis=1)
        large_exponents_index = np.argmax(exponents >= 20.0, axis=1)

        # Not quite sure how to do this elegantly without a for loop,
        # but at this point it doesn't really matter too much.
        # Most of the computationally intensive work has already been done.
        for r in range(0, X.shape[0]):
            if large_exponents[r]:
                probabilities[r, large_exponents_index[r]] = 1.0 # Rest of this row is kept as zero.
                continue

            # No large exponents. Things are proportionla to (class probability) * e^(exponent)
            sizes = np.exp(exponents[r,:]) * self.outcome_fractions
            total_size = sizes.sum()
            
            probabilities[r,:] = sizes / total_size

        return probabilities

    
    def debug_func(self):
        print("Means")
        for key in self.means:
            print(self.means[key])

        print()
        print("Outcomes (classes)")
        print(self.outcomes)

        print()
        print("Outcome fractions")
        print(self.outcome_fractions)

        print()
        print("The underlying variance matrix")
        print(self.variance)

        print()
        print("The underlying scaler")
        print(self.error_scaler)

        print()
        print("Eigenvalues of the underlying variance matrix")
        print(np.linalg.eig(self.variance)[0])