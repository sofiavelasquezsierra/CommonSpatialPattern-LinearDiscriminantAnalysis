import numpy as np

def lda_fit(X, y, reg=1e-4):
    """
    Two-class LDA with shared covariance. y must be binary labels {c1, c2}.
    Returns LDAParams with projection w and bias b.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("LDA expects exactly two classes.")
    c1, c2 = classes[0], classes[1]
    X1 = X[y == c1]
    X2 = X[y == c2]
    mu1 = X1.mean(axis=0)
    mu2 = X2.mean(axis=0)
    # Pooled covariance
    S1 = np.cov(X1, rowvar=False)
    S2 = np.cov(X2, rowvar=False)
    n1, n2 = len(X1), len(X2)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2) 
    # Regularize
    Sp += reg * np.eye(Sp.shape[0])
    # Solve for w = Σ^{-1}(μ2 - μ1)
    w = np.linalg.solve(Sp, (mu2 - mu1))
    # Bias using equal priors
    b = -0.5 * (mu1 + mu2) @ w
    return w, float(b), c1, c2

def lda_predict(w, b, c1, c2, X):
    s = X @ w + b # score each sample based on the calculated class boundary
    # Map sign of s to predicted labels
    yhat = np.where(s > 0, c2, c1)
    return yhat