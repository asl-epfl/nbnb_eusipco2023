import numpy as np

def gaussian(x, m, var):
    '''
    Computes the Gaussian pdf value at x.
    Arguments
    ---------
    x: ndarray
        value at which the pdf is computed
    m: float
        mean
    var: float
        variance
    
    Returns
    -------
    p: ndarray
        pdf evaluated at x
    '''
    p = np.exp(-(x-m)**2/(2*var))/(np.sqrt(2*np.pi*var))
    return p


def laplace(x, m, b):
    '''
    Computes the Laplace pdf value at x.
    
    Arguments
    ---------
    x: ndarray
        value at which the pdf is computed
    m: float
        mean
    b: float
        scale parameter
    
    Returns
    -------
    p: ndarray
        pdf evaluated at x
    '''
    p = np.exp(-np.abs((x-m)/b))/(2*b)
    return p


def sl_bayesian_update_batch(L, mu, gamma):
    '''
    Computes the Bayesian update for a batch of samples.
    Data is input as tensors of dimension N_MC x K x H, where 
    N_MC is the number of Monte Carlo repetitions, 
    K is the number of agents and H the number of hypotheses.
    
    Arguments
    ---------
    L: ndarray
        likelihoods tensor (N_MC x K x H)
    mu: ndarray
        beliefs tensor (N_MC x K x H)
    gamma: ndarray
        vector of weights (K x 1)
    
    Returns
    -------
    bu: ndarray
        Bayesian update
    '''
    aux = L**gamma[None, :, None] * mu
    bu = aux/aux.sum(axis = 2)[:, :, None]
    return bu


def DKL(m,n,dx):
    '''
    Computes the KL divergence between vectors m and n.
    
    Arguments
    ---------
    m: ndarray
        true distribution in vector form
    n: ndarray
        second distribution in vector form
    dx : float 
        sample size
        
    Returns
    -------
    _ : float
        KL divergence
    '''
    mn = m/n
    mnlog = np.log(mn)
    return np.sum(m*dx*mnlog)


def lmgf(thetazero, theta1, t):
    '''
    Computes the LMGF for the family of Laplace distributions
    
    Arguments
    ---------
    thetazero: float
        true state of nature
    theta1:  float
        state different than true state
    t: ndarray
        vector of real values
    
    Returns
    -------
    f: ndarray
        vector of values of LMGF over t
    '''
    alpha = theta1 - thetazero
    if thetazero >= theta1:
        f = np.log(np.exp(alpha * (t + 1)) + np.exp(-alpha * t) - np.exp(alpha / 2)
                   * np.sinh(alpha*(t + 1/2))/(t + 1/2)) - np.log(2)
    elif thetazero<theta1:
        f = np.log(np.exp(-alpha * (t + 1)) + np.exp(alpha * t) + np.exp(-alpha / 2)
                   * np.sinh(alpha*(t + 1/2))/(t + 1/2)) - np.log(2)
    return f


def sl_batch(mu_0, csi, A, N_ITER, theta, var, is_gaussian = False, gamma = []):
    '''
    Runs the social learning algorithm with Gaussian or Laplace likelihoods for a batch of samples.
    Data is input as tensors of different dimensions, where N_MC is the number of Monte Carlo repetitions, 
    K is the number of agents and H the number of hypotheses, N_ITER is the number of iterations.
    
    Arguments
    ---------
    mu_0: ndarray
        initial beliefs (N_MC x K x H)
    csi: ndarray
        observations (N_MC x K x N_ITER)
    A: ndarray
        Combination matrix (K x K)
    N_ITER: int
        number of iterations
    theta: ndarray
        vector of means for the likelihoods (H x 1)
    var: float
        variance of Gaussian likelihoods/ scale parameter of the Laplace likelihoods
    is_gaussian: bool
        flag indicating if the likelihoods are Gaussian
    gamma: ndarray
        vector of gamma weights (K x 1)
        
    Returns
    -------
    MU: ndarray
        beliefs tensor (N_ITER +1 x N_MC x K x H)
    '''    
    mu = np.tile(mu_0,[csi.shape[0], 1, 1])
    N = len(A)
    MU = [mu]
    for i in range(N_ITER):
        if is_gaussian:
            L_i = np.array([gaussian(csi[:, :, i], t, var).T for t in theta]).T
        else:
            L_i = np.array([laplace(csi[:, :, i], t, var).T for t in theta]).T
        
        # unindentifiability
        L_i[:, :N//3, 1] = L_i[:, :N//3, 0]
        L_i[:, N//3:2*N//3, 1] = L_i[:, N//3:2*N//3, 2]
        L_i[:, 2*N//3:, 2] = L_i[:, 2*N//3:, 0]
        
        if len(gamma)>0:    
            psi = sl_bayesian_update_batch(L_i, mu, gamma)
        else:
            psi = sl_bayesian_update_batch(L_i, mu, np.ones(len(A)))
        decpsi = np.log(psi)
        mu = np.exp((A.T)@(decpsi)) / np.sum(np.exp((A.T)@(decpsi)), axis = 2)[:, :, None]
        MU.append(mu)

    return np.array(MU)


def sl_batch2(mu_0, csi, A, N_ITER, theta, var, is_gaussian = False, gamma = []):
    '''
    Runs the social learning algorithm with Gaussian or Laplace likelihoods for a batch of samples,
    for the clustered example.
    Data is input as tensors of different dimensions, where N_MC is the number of Monte Carlo repetitions, 
    K is the number of agents and H the number of hypotheses, N_ITER is the number of iterations.
    
    Arguments
    ---------
    mu_0: ndarray
        initial beliefs (N_MC x K x H)
    csi: ndarray
        observations (N_MC x K x N_ITER)
    A: ndarray
        Combination matrix (K x K)
    N_ITER: int
        number of iterations
    theta: ndarray
        vector of means for the likelihoods (H x 1)
    var: float
        variance of Gaussian likelihoods/ scale parameter of the Laplace likelihoods
    is_gaussian: bool
        flag indicating if the likelihoods are Gaussian
    gamma: ndarray
        vector of gamma weights (K x 1)
        
    Returns
    -------
    MU: ndarray
        beliefs tensor (N_ITER +1 x N_MC x K x H)
    '''    
    mu = np.tile(mu_0,[csi.shape[0], 1, 1])
    N = len(A)
    MU = [mu]
    for i in range(N_ITER):
        if is_gaussian:
            L_i = np.array([gaussian(csi[:, :, i], t, var).T for t in theta]).T
        else:
            L_i = np.array([laplace(csi[:, :, i], t, var).T for t in theta]).T
            L_i[:, 1:] = np.array([laplace(csi[:, 1:, i], t*0.5, var).T for t in theta]).T
                
        if len(gamma)>0:    
            psi = sl_bayesian_update_batch(L_i, mu, gamma)
        else:
            psi = sl_bayesian_update_batch(L_i, mu, np.ones(len(A)))
        decpsi = np.log(psi)
        mu = np.exp((A.T)@(decpsi)) / np.sum(np.exp((A.T)@(decpsi)), axis = 2)[:, :, None]
        MU.append(mu)

    return np.array(MU)


def isGraphStronglyConnected(G):
    '''
    Checks whether the adjacency matrix G corresponds to a strongly connected graph
    
    Arguments
    ---------
    G: ndarray
        adjacency matrix 
    
    Returns
    -------
    _: bool
        strongly connected flag
    '''
    eigG, eigvG = np.linalg.eig(G)
    eigone = np.argmax(eigG)
    perron = eigvG[:, eigone]
    perron = perron / np.sum(perron)
    return np.all(perron > 0), np.real(perron)

def nudge(pos, x_shift, y_shift):
    '''
    Perturbs a dictionary of (x,y) positions by some fixed shift values.
    
    Arguments
    ---------
    pos: dict(tuple(floats))
        dictionary of positions
    x_shift: float
        x-variable shift constant
    y_shift: float
        y-variable shift constant
    
    Returns
    -------
    _: dict(tuple(floats))
        shifted dictionary
    '''
    return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

def compute_error_exp(t0, pv, id_matrix, theta, t):
    '''
    Compute the theoretical error exponent of the probability of error
    for the first simulation setting under a fixed true state t0.
    
    Arguments
    ---------
    t0: int
        true state
    pv: ndarray
        Perron vector of network
    id_matrix: ndarray
        identifiability matrix
    theta: ndarray
        vector of distribution means
    t: ndarray
        grid used for the computation of the LMGF
    
    Returns
    -------
    _: float
        dominant error exponent
    '''
    t = np.linspace(-30, 30, 10000)
    w, phi = [], []
    for i in range(len(theta)):
        if i!= t0:
            w.append(np.sum(np.array([lmgf(theta[id_matrix[0, t0]], theta[id_matrix[0, i]], pk * t) for pk in pv[:3]] + 
                            [lmgf(theta[id_matrix[3, t0]], theta[id_matrix[3, i]], pk * t) for pk in pv[3:6]] +
                            [lmgf(theta[id_matrix[6, t0]], theta[id_matrix[6, i]], pk * t) for pk in pv[6:]]), axis=0))
        
    # w = np.sum(np.array([lmgf(theta[id_matrix[0, t0]], theta[id_matrix[0, t1]], pk * t) for pk in pv[:3]] + 
    #                     [lmgf(theta[id_matrix[3, t0]], theta[id_matrix[3, t1]], pk * t) for pk in pv[3:6]] +
    #                     [lmgf(theta[id_matrix[6, t0]], theta[id_matrix[6, t1]], pk * t) for pk in pv[6:]]), axis=0)
    # w2 = np.sum(np.array([lmgf(theta[id_matrix[0, t0]], theta[id_matrix[0, t2]], pk * t) for pk in pv[:3]] +
    #                      [lmgf(theta[id_matrix[3, t0]], theta[id_matrix[3, t2]], pk * t) for pk in pv[3:6]] +
    #                      [lmgf(theta[id_matrix[6, t0]], theta[id_matrix[6, t2]], pk * t) for pk in pv[6:]]), axis=0)
            phi.append(-min(w[-1]))
    # phi1 = -min(w)
    # phi2 = -min(w2)
    # phi = min(-min(w), -min(w2))
    return min(phi)
  
    
# def compute_error_exp_1(pv):
#     t = np.linspace(-30, 30, 10000)
#     w = np.sum(np.array([lmgf(theta[id_matrix[0, 1]], theta[id_matrix[0, 1]], pk * t)  for pk in pv[:3]] + 
#                         [lmgf(theta[id_matrix[3, 1]], theta[0], pk * t) for pk in pv[3:6]]+
#                         [lmgf(theta[id_matrix[6, 1]], theta[0], pk * t) for pk in pv[6:]]), axis=0)
#     w2 = np.sum(np.array([lmgf(theta[id_matrix[0, 1]], theta[2], pk * t) for pk in pv[:3]]+
#                          [lmgf(theta[id_matrix[3, 1]], theta[2], pk * t) for pk in pv[3:6]]+
#                          [lmgf(theta[id_matrix[6, 1]], theta[0], pk * t) for pk in pv[6:]]), axis=0)
#     phi1 = -min(w)
#     phi2 = -min(w2)
#     phi = min(-min(w), -min(w2))
#     return phi, phi1, phi2
  
    
# def compute_error_exp_2(pv):
#     t = np.linspace(-30, 30, 10000)
#     w = np.sum(np.array([lmgf(theta[id_matrix[0, 2]], theta[0], pk * t)  for pk in pv[:3]] + 
#                         [lmgf(theta[id_matrix[3, 2]], theta[0], pk * t) for pk in pv[3:6]] +
#                         [lmgf(theta[id_matrix[6, 2]], theta[0], pk * t) for pk in pv[6:]]), axis=0)
#     w2 = np.sum(np.array([lmgf(theta[id_matrix[0, 2]], theta[0], pk * t) for pk in pv[:3]] +
#                          [lmgf(theta[id_matrix[3, 2]], theta[2], pk * t) for pk in pv[3:6]] +
#                          [lmgf(theta[id_matrix[6, 1]], theta[0], pk * t) for pk in pv[6:]]), axis=0)
#     phi1 = -min(w)
#     phi2 = -min(w2)
#     phi = min(-min(w), -min(w2))
#     return phi, phi1, phi2
  
    
# def compute_error_exp22(pv):
#     t = np.linspace(-30, 30, 10000)
#     w = np.sum(np.array([lmgf(theta[2], theta[0], pk * t) for pk in pv[[0]]]+
#                        [lmgf(0.5*theta[2], 0.5*theta[0], pk * t) for pk in pv[1:]]), axis=0)
#     w2 = np.sum(np.array([lmgf(theta[2], theta[1], pk * t) for pk in pv[[0]]]+
#                        [lmgf(0.5*theta[2], 0.5*theta[1], pk * t) for pk in pv[1:]]), axis=0)
#     phi1 = -min(w)
#     phi2 = -min(w2)
#     phi = min(-min(w), -min(w2))
#     return phi, phi1, phi2

# def compute_error_exp21(pv):
#     t = np.linspace(-30, 30, 10000)
#     w = np.sum(np.array([lmgf(theta[1], theta[0], pk * t) for pk in pv[[0]]]+
#                        [lmgf(0.5*theta[1], 0.5*theta[0], pk * t) for pk in pv[1:]]), axis=0)
#     w2 = np.sum(np.array([lmgf(theta[1], theta[2], pk * t) for pk in pv[[0]]]+
#                        [lmgf(0.5*theta[1], 0.5*theta[2], pk * t) for pk in pv[1:]]), axis=0)
#     phi1 = -min(w)
#     phi2 = -min(w2)
#     phi = min(-min(w), -min(w2))
#     return phi, phi1, phi2

def compute_error_exp_cluster(t0, pv, theta, t):
    '''
    Compute the theoretical error exponent of the probability of error
    for the second simulation setting under a fixed true state t0,
    and two clusters {1} and {2,3,..., N}
    
    Arguments
    ---------
    t0: int
        true state
    pv: ndarray
        Perron vector of network
    theta: ndarray
        vector of distribution means
    t: ndarray
        grid used for the computation of the LMGF
    
    Returns
    -------
    _: float
        dominant error exponent
    '''
    w, phi = [], []
    for i in range(len(theta)):
        if i!= t0:
            w.append(np.sum(np.array([lmgf(theta[t0], theta[i], pk * t) for pk in pv[[0]]]+
                               [lmgf(0.5*theta[t0], 0.5*theta[i], pk * t) for pk in pv[1:]]), axis=0))
    # w2 = np.sum(np.array([lmgf(theta[t0], theta[2], pk * t) for pk in pv[[0]]]+
    #                    [lmgf(0.5*theta[t0], 0.5*theta[2], pk * t) for pk in pv[1:]]), axis=0)
            phi.append(-min(w[-1]))
    # phi1 = -min(w)
    # phi2 = -min(w2)
    # phi = min(-min(w), -min(w2))
    return min(phi)


