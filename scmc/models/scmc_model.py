import scipy.stats
import numpy as np


def scmc(A, D, Lambda, v, alpha, beta, max_iter=10000, min_iter=1000):
    """Gibbs sampler for mutation correlation model.

    Estimate covariance matrix (up to a constant), genotype and read-error / 
    allelic imbalance parameters.

    Args:
        A : 2-D array_like, (d, n)
            Number of reads mapping to alternative allele.
        D : 2-D array_like, (d, n)
            Total number of reads mapping to genomic position.
        Lambda : 2-D array_like (d, d)
            Inverse-Wishart scale matrix for covariance prior (positive definite).
        v : float
            Inverse-Wishart degrees of freedom for covariance prior (v > p-1).
        alpha : 1-D array_like (d+1, )
            Beta prior parameters.
        beta : 1-D array_like (d+1, )
            Beta prior parameters.
        max_iter : int
            Maximum number of iterations.
        min_iter : int
            Minimum number of iterations.

    Returns:
        theta : 1-D array_like (d+1, )
            Estimated read-error / allelic imbalance parameters.
        Sigma : 2-D array_like (d, d)
            Estimated covariance matrix.
        X : 2-D array_like, (d, n)
            Estimated genotype (0 - homozygous reference, 1 - heterozygous).
    """
    [d, n] = A.shape

    theta_all = np.zeros([d+1, max_iter])
    Sigma_all = np.zeros([d, d, max_iter])
    X_all = np.zeros([d, n, max_iter])

    # initialize
    theta_all[:, 0] = scipy.stats.beta.rvs(a=alpha, b=beta)
    X_all[:, :, 0] = (A/(D + 1e-5) > 0.1).astype(int)
    Sigma_all[:, :, 0] = sample_Sigma(X_all[:, :, 0], Lambda, v)

    for it in range(1, max_iter):
        if it % 100 == 0:
            print(it)
        theta_all[:, it] = sample_theta(A, D, X_all[:, :, it-1], alpha, beta)
        Sigma_all[:, :, it] = sample_Sigma(X_all[:, :, it-1], Lambda, v)
        X_all[:, :, it] = sample_X(Sigma_all[:, :, it], X_all[:, :, it-1], A, D, theta_all[:, it])

    it_burnin = np.ceil(it * 0.25).astype(int) - 1

    theta = theta_all[:, it_burnin:it].mean(1)
    Sigma = Sigma_all[:, :, it_burnin:it].mean(2)
    X = X_all[:, :, it_burnin:it].mean(2)

    return theta, Sigma, X


def sample_theta(A, D, X, alpha, beta):
    """Sample theta from binomial distributions."""
    U = (X > 0).astype(int)
    u_0 = (A*(1-U)).sum()
    v_0 = ((D-A)*(1-U)).sum()

    u = np.append(u_0, (A*U).sum(1))
    v = np.append(v_0, ((D-A)*U).sum(1))

    u = u + alpha
    v = v + beta

    return scipy.stats.beta.rvs(a=u, b=v)


def sample_X(Sigma, X, A, D, theta):
    """Sample X from Bernoulli-truncated normal."""
    p_smaller_zero = scipy.stats.binom.pmf(A, D, theta[0])
    p_greater_zero = scipy.stats.binom.pmf(A, D, theta[1:, np.newaxis])

    [d, n] = X.shape

    Z = np.copy(X)

    for i in range(d):
        ids = list(range(d))
        ids.pop(i)

        # TODO standardize distribution beforehand to speed up computations?

        S_ids_inv = np.linalg.inv(Sigma[np.ix_(ids, ids)])
        Q = np.matmul(Sigma[i, ids], S_ids_inv)
        sigma_ij = np.sqrt(Sigma[i, i] - np.matmul(Q, Sigma[ids, i]))

        for j in range(n):
            mu_ij = np.matmul(Q, Z[ids, j])

            # sample region to truncate
            p_S = scipy.stats.norm.sf(0, loc=mu_ij, scale=sigma_ij)
            c_S = p_greater_zero[i, j]
            c_S_complement = p_smaller_zero[i, j]
            p = c_S * p_S / (c_S * p_S + c_S_complement * (1 - p_S))

            if np.isnan(p):
                print(p_S)

            # sample from truncated normal
            if scipy.stats.bernoulli.rvs(p) > 0.5:
                z = sample_truncated_normal(0, np.inf, loc=mu_ij, scale=sigma_ij)
            else:
                z = sample_truncated_normal(-np.inf, 0, loc=mu_ij, scale=sigma_ij)

            Z[i, j] = z
    return Z


def sample_truncated_normal(a, b, loc=0, scale=1):
    """Sample from truncated normal distribution."""
    a, b = (a - loc) / scale, (b - loc) / scale
    return scipy.stats.truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)


def sample_Sigma(X, Lambda, v):
    """Sample Sigma from inverse Wishart distribution."""
    n = X.shape[1]
    return scipy.stats.invwishart.rvs(df=v + n, scale=Lambda + np.matmul(X, X.transpose()))


def generate_from_prior(D, Lambda, v, alpha, beta):
    """Generate data from prior."""
    [d, n] = D.shape

    theta = scipy.stats.beta.rvs(a=alpha, b=beta)
    Sigma = scipy.stats.invwishart.rvs(df=v, scale=Lambda)
    X = scipy.stats.multivariate_normal.rvs(np.zeros(d), Sigma, n).transpose()

    U = (X > 0).astype(int)
    A = generate_reads(U, D, theta)

    return theta, Sigma, X, A


def generate_reads(U, D, theta):
    """Generate alternative reads for given coverage matrix D, 
    genotype U and parameter theta."""
    A0 = scipy.stats.binom.rvs(D, theta[0])
    Ai = scipy.stats.binom.rvs(D, theta[1:, np.newaxis])

    A = U * Ai + (1 - U) * A0

    return A


def Geweke_Z(X, first=0.1, last=0.5):
    """Geweke diagnostics for MCMC chain convergence.

    See Geweke J. Evaluating the accuracy of sampling-based approaches to the
    calculation of posterior moments[M]. Minneapolis, MN, USA: Federal Reserve
    Bank of Minneapolis, Research Department, 1991.
    and https://pymc-devs.github.io/pymc/modelchecking.html#formal-methods

    Args:
        X : 1-D array_like, (n, )
            The uni-variate MCMC sampled chain for convergence diagnostic.
        first : float
            The proportion of first part in Geweke diagnostics.
        last : float
            The proportion of last part in Geweke diagnostics.

    Returns:
        Z : float
            The Z score of Geweke diagnostics.
    """

    n = X.shape[0]
    A = X[:int(first*n)]
    B = X[int(last*n):]
    if np.sqrt(np.var(A) + np.var(B)) == 0:
        Z = None
    else:
        Z = abs(A.mean() - B.mean()) / np.sqrt(np.var(A) + np.var(B))
    return Z


# def elliptical_slice(init_state, L, loglik, init_loglik=None):
#     """Perform elliptical slice sampling.

#     https://arxiv.org/abs/1001.0175

#     Args:
#         init_state : 1-D array_like, (d, )
#             The current state vector.
#         L : 2-D array_like (d, d)
#             Lower triangular Cholesky factor (numpy.linalg.cholesky)
#         loglik : function
#             Log-Likelihood function.
#         init_loglik : float
#             The value of loglik at init_state (optional).

#     Returns
#         new_state : 1-D array_like, (d, )
#             The new state vector.
#         new_loglik : float
#             The value of loglik at new_state.
#     """

#     # TODO error checking

#     d=init_state.shape[0]

#     # Step 1: Choose ellipse
#     nu=np.dot(L, np.random.normal(size=d))

#     # Step 2: Log-likelihood threshold
#     if init_loglik is None:
#         init_loglik=loglik(init_state)
#     loglik_threshold=init_loglik + np.log(np.random.uniform())

#     # Step 3: Initial proposal
#     theta=np.random.uniform() * 2 * np.pi
#     theta_min=theta - 2 * np.pi
#     theta_max=theta

#     # Step 4-10: Find slice
#     while True:
#         new_state=init_state * np.cos(theta) + nu * np.sin(theta)
#         new_loglik=loglik(new_state)

#         if new_loglik > loglik_threshold:
#             break

#         # shrink bracket
#         if theta < 0:
#             theta_min=theta
#         else:
#             theta_max=theta

#         theta=np.random.uniform() * (theta_max - theta_min) + theta_min

#     return new_state, new_loglik
