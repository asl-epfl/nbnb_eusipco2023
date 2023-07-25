import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import os
from functions import *
from cycler import cycler 

color_list_df = ['#2274A5', 
                 '#5FBB97', 
                 '#DA462F', 
                 '#FFC847', 
                 '#B045A9']

mpl.style.use('seaborn-deep')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 16
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
mpl.rcParams['axes.prop_cycle'] = cycler(color=color_list_df)
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.2

# Figure path
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)

# Setup
N = 10
M = 3
np.random.seed(0)

# Graph
G = np.random.choice([0.0, 1.0],
                     size=(N, N), 
                     p=[0.7, 0.3])
G = G.T + G
G = G + np.eye(N)
G = (G > 0) * 1.0

G[9,:9] = 0.
G[:9, 9] = 0.
G[9, 5] = 1.
G[5, 9] = 1.

# Metropolis Rule -> Doubly-stochastic
A = np.zeros((N, N))
deg = np.sum(G, 0)
degmax = max(deg)
for j in range(N):
    for i in range(N):
        if G[i, j] > 0:
            A[i, j] = 1 / max(deg[i], deg[j])

    A[j, j] = 1 - np.sum(A[:j, j]) - np.sum(A[j+1:,j])

st, p = isGraphStronglyConnected(A)
print('Graph is strongly connected:', st)

A_fully = np.ones((N, N)) / N
A_ls = G/np.sum(G, 0)

Gr = nx.from_numpy_array(A)
pos = nx.kamada_kawai_layout(Gr)

label_pos = nudge(pos, 0.06, 0.07)

# Plot Network topology
f, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.axis('off')
ax.set_ylim([-1.1,.7])
ax.set_xlim([-.7,1.])

nx.draw_networkx_labels(Gr, 
                        pos=label_pos, 
                        labels={i: i+1 for i in range(N)}, 
                        bbox=dict(boxstyle='round', 
                                  fc='white', 
                                  ec='None', 
                                  pad=0, 
                                  alpha=0.9), 
                        font_size=18)


nx.draw_networkx_nodes(Gr, 
                       pos=pos, 
                       node_color='C1',
                       nodelist=range(0, N), 
                       node_size=70,  
                       linewidths=0.5,
                       ax=ax)

nx.draw_networkx_edges(Gr, 
                       pos=pos, 
                       node_size=100, 
                       alpha=1, 
                       arrowsize=6, 
                       width=0.5);
f.savefig(FIG_PATH + 'fig1_left_a.pdf', bbox_inches='tight')

Gr = nx.from_numpy_array(A)
pos = nx.kamada_kawai_layout(Gr)

label_pos = nudge(pos, 0.06, 0.07)

f, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.axis('off')
ax.set_ylim([-1.1,.7])
ax.set_xlim([-.7,1.])

nx.draw_networkx_labels(Gr, 
                        pos=label_pos, 
                        labels={i: i+1 for i in range(N)}, 
                        bbox=dict(boxstyle='round', 
                                  fc='white', 
                                  ec='None', 
                                  pad=0, 
                                  alpha=0.9), 
                        font_size=18)

nx.draw_networkx_nodes(Gr, 
                       pos=pos, 
                       node_color='C2',
                       nodelist=[0], 
                       node_size=70,  
                       linewidths=0.5,
                       ax=ax)


nx.draw_networkx_nodes(Gr, 
                       pos=pos, 
                       node_color='C0',
                       nodelist=range(1, N), 
                       node_size=70,  
                       linewidths=0.5,
                       ax=ax)

nx.draw_networkx_edges(Gr, 
                       pos=pos, 
                       node_size=100, 
                       alpha=1, 
                       arrowsize=6, 
                       width=0.5);
f.savefig(FIG_PATH + 'fig1_right_a.pdf', bbox_inches='tight')

# Hypotheses
theta = np.arange(1,4) * 0.1
b = 1
x = np.linspace(-10, 10, 1000)
dt = (max(x) - min(x)) / len(x)

# Likelihoods
L0 = laplace(x, theta[0], b)
L1 = laplace(x, theta[1], b)
L2 = laplace(x, theta[2], b)
L = np.array([L0, L1, L2])

# Initialization
np.random.seed(0)
mu_0 = np.random.rand(N, M)
mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]


##################### Left-stochastic matrix #####################
# First simulation setting, where we compare the performance of 
# Centralized Bayes, Social Learning and NB^2 strategy using the 
# knowledge of the Perron vector of the network.
##################################################################

# Compute Perron vectors for all combination matrices
# DS matrix
eigval_A, eigvec_A = np.linalg.eig(A)
pv_A = eigvec_A[:, np.isclose(eigval_A, 1.)]
pv_A = pv_A/np.sum(pv_A)

# Fully connected network
eigval_fully, eigvec_fully = np.linalg.eig(A_fully)
pv_fully = eigvec_fully[:, np.isclose(eigval_fully, 1.)]
pv_fully = pv_fully/np.sum(pv_fully)

# LS matrix
eigval_ls, eigvec_ls = np.linalg.eig(A_ls)
pv_ls = eigvec_ls[:, np.isclose(eigval_ls, 1.)]
pv_ls = pv_ls / np.sum(pv_ls)

# identifiability matrix
id_matrix = np.array([[0, 0, 2],
                [0, 0, 2],
                [0, 0, 2],
                [0, 2, 2],
                [0, 2, 2],
                [0, 2, 2],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]])

np.random.seed(0)
N_MC = 50000
N_ITER = 400

# Run Monte Carlo simulations
csi, th_vec = [], []
for j in range(N_MC):
    th = np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])
    th_vec.append(th)
    csi_ag = []
    for l in range(N):
        true_dist = id_matrix[l, th]
        csi_ag.append(np.random.laplace(theta[true_dist], b, size=(N_ITER)))
    csi.append(csi_ag)
csi = np.array(csi)

MU_fully = sl_batch(mu_0, 
                    csi, 
                    A_fully, 
                    N_ITER, 
                    theta, 
                    b, 
                    is_gaussian=False)
MU_ls = sl_batch(mu_0, 
                 csi, 
                 A_ls, 
                 N_ITER, 
                 theta, 
                 b, 
                 is_gaussian=False)
MU_ls_gamma = sl_batch(mu_0, 
                       csi, 
                       A_ls, 
                       N_ITER, 
                       theta, 
                       b, 
                       is_gaussian=False, 
                       gamma=1/pv_ls[:, 0])


# Compute theoretical error exponents
t = np.linspace(-30, 30, 10000)

phi_fully0 = compute_error_exp(0, 
                               pv_fully, 
                               id_matrix, 
                               theta, 
                               t)
phi_ls0 = compute_error_exp(0, 
                            pv_ls, 
                            id_matrix, 
                            theta, 
                            t)

phi_fully1 = compute_error_exp(1, 
                                 pv_fully, 
                                 id_matrix, 
                                 theta, 
                                 t)
phi_ls1 = compute_error_exp(1, 
                              pv_ls, 
                              id_matrix, 
                              theta, 
                              t)

phi_fully2 = compute_error_exp(2, 
                                 pv_fully, 
                                 id_matrix, 
                                 theta, 
                                 t)
phi_ls2 = compute_error_exp(2, 
                              pv_ls, 
                              id_matrix, 
                              theta, 
                              t)

phi_fully = min([phi_fully0, phi_fully1, phi_fully2])
phi_ls = min([phi_ls0, phi_ls1, phi_ls2])

# Compute empirical probabilities
p_fully = np.sum(np.argmax(MU_fully, axis=3)!=np.array(th_vec)[None, :, None], axis=1) / N_MC
p_ls = np.sum(np.argmax(MU_ls, axis=3)!=np.array(th_vec)[None, :, None], axis=1) / N_MC
p_ls_gamma = np.sum(np.argmax(MU_ls_gamma, axis=3)!=np.array(th_vec)[None, :, None], axis=1) / N_MC

p_fully_net = p_fully.mean(1)
p_ls_net = p_ls.mean(1)
p_ls_gamma_net = p_ls_gamma.mean(1)

# Plot of probability curves
# Compute approximations
# Large deviations approximation
iy = np.arange(0, N_ITER + 1, 20)
y_fully = np.exp(-phi_fully * iy)
y_ls = np.exp(-phi_ls * iy)
n_vector = np.arange(0, N_ITER + 1, 28)

f, ax = plt.subplots(1, 1, figsize=(6,4))
ax.plot(iy, 0.2 * y_fully, 
        color='k', 
        linewidth=2, 
        linestyle='dashed', 
        zorder=0)
ax.plot(iy, 0.28 * y_ls, 
        color='red', 
        linewidth=2, 
        linestyle='dashed', 
        zorder=0)

ax.set_yscale('log')
ax.set_xlim(0, 400)
ax.set_ylim([1e-4, 1])
ax.set_xlabel(r'$i$', fontsize=16)
ax.set_ylabel(r'$p_{i}$', fontsize=16)

h = []
h.append(ax.scatter(n_vector + 1, 
                    p_ls_gamma_net[n_vector], 
                    marker='o', 
                    color='C0', 
                    s=70, 
                    linewidth=2, 
                    facecolor='None'))
h.append(ax.scatter(n_vector + 1, 
                    p_ls_net[n_vector], 
                    marker='o', 
                    color='C2', 
                    s=70, 
                    linewidth=2, 
                    facecolor='None'))
h.append(ax.scatter(n_vector + 1, 
                    p_fully_net[n_vector], 
                    marker='x', 
                    color='k', 
                    s=30, 
                    linewidth=2))
ax.legend(h,[r'NB$^2$', 'SL', 'Bayes'], 
          fontsize=14, 
          loc='upper right', 
          bbox_to_anchor=(.6, 1.1), 
          framealpha=1)
ax.text(8, 1.6e-4, 
        'Markers: Simulation\n Lines: Theoretical exponents', 
        bbox=dict(facecolor='white', alpha=0.5))
f.savefig(FIG_PATH + 'fig1_left_b.pdf', bbox_inches='tight')

################### Highly-dependent clusters ###################
# Second simulation setting, where we compare the performance of 
# Centralized Bayes, Social Learning and NB^2 strategy using the 
# knowledge of clusters of highly dependent agents.
#################################################################
theta = np.arange(1,4) * 0.1
vec = np.array([1, 9, 9, 9, 9, 9, 9, 9, 9, 9]) 

# Highly correlated rho = 2/3
np.random.seed(0)
N_MC = 10000
N_ITER = 1200

#%% Run Monte Carlo simulations
csi2, th_vec2 = [], []
for j in range(N_MC):
    th2 = np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])
    th_vec2.append(th2)
    csi_ag2 = []
    dum2 = np.random.laplace(0.5 * theta[th2], b, size=(N_ITER))

    for l in range(N):
        if l == 0:
            csi_ag2.append(np.random.laplace(theta[th2], b, size=(N_ITER)))
        else:
            csi_ag2.append(dum2.copy() + np.random.normal(0, 1, size=(N_ITER)))
    csi2.append(csi_ag2)
csi2 = np.array(csi2)

# Totally correlated rho = 1
np.random.seed(0)
N_MC = 10000
N_ITER = 1200

#%% Run Monte Carlo simulations
csi, th_vec = [], []
for j in range(N_MC):
    th = np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])
    th_vec.append(th)
    csi_ag = []
    dum = np.random.laplace(0.5 * theta[th], b, size=(N_ITER))

    for l in range(N):
        if l == 0:
            csi_ag.append(np.random.laplace(theta[th], b, size=(N_ITER)))
        else:
            csi_ag.append(dum.copy())
    csi.append(csi_ag)
csi = np.array(csi)

# Centralized Bayes
A_Bay = np.ones((2,2))/2
MU_fully = sl_batch2(mu_0[:2], 
                     csi[:,:2,:], 
                     A_Bay, 
                     N_ITER, 
                     theta, 
                     b, 
                     is_gaussian=False)

# Compute theoretical error exponent for the centralized Bayes
phi_fully0 = compute_error_exp_cluster(0, 
                                       np.ones(2)/2, 
                                       theta, 
                                       t)
phi_fully1 = compute_error_exp_cluster(1, 
                                       np.ones(2)/2, 
                                       theta, 
                                       t)
phi_fully2 = compute_error_exp_cluster(2, 
                                       np.ones(2)/2, 
                                       theta, 
                                       t)

phi_fully = min([phi_fully0, phi_fully1, phi_fully2])

# Social learning and NB^2 for rho = 2/3
MU_A2 = sl_batch2(mu_0, 
                  csi2, 
                  A, 
                  N_ITER, 
                  theta, 
                  b, 
                  is_gaussian=False)

MU_A_gamma2 = sl_batch2(mu_0, 
                        csi2, 
                        A, 
                        N_ITER, 
                        theta, 
                        b, 
                        is_gaussian=False, 
                        gamma=1/(pv_A[:, 0]*vec))

# Social learning and NB^2 for rho = 1
MU_A = sl_batch2(mu_0, 
                 csi, 
                 A, 
                 N_ITER, 
                 theta, 
                 b, 
                 is_gaussian=False)

MU_A_gamma = sl_batch2(mu_0, 
                       csi, 
                       A, 
                       N_ITER, 
                       theta, 
                       b, 
                       is_gaussian=False, 
                       gamma=1/(pv_A[:, 0]*vec))

# Compute empirical probabilities
p_fully = np.sum(np.argmax(MU_fully, axis=3)!=np.array(th_vec)[None, :, None], axis=1) / N_MC
p_A = np.sum(np.argmax(MU_A, axis=3)!=np.array(th_vec)[None, :, None], axis=1) / N_MC
p_A_gamma = np.sum(np.argmax(MU_A_gamma, axis=3)!=np.array(th_vec)[None, :, None], axis=1) / N_MC
p_A2 = np.sum(np.argmax(MU_A2, axis=3)!=np.array(th_vec2)[None, :, None], axis=1) / N_MC
p_A_gamma2 = np.sum(np.argmax(MU_A_gamma2, axis=3)!=np.array(th_vec2)[None, :, None], axis=1) / N_MC

p_fully_net = p_fully.mean(1)
p_A_net = p_A.mean(1)
p_A_gamma_net = p_A_gamma.mean(1)
p_A2_net = p_A2.mean(1)
p_A_gamma2_net = p_A_gamma2.mean(1)

# Plot of probability curves
# Compute approximations
# Large deviations approximation
iy = np.arange(0, N_ITER + 1, 20)
y_fully = np.exp(-phi_fully * iy)
n_vector = np.arange(0, N_ITER + 1, 80)

#%% Plot probability of error
f, ax = plt.subplots(1, 1, figsize=(6,4))
ax.plot(iy, 0.25 * y_fully, 
        color='k', 
        linewidth=2, 
        linestyle='dashed', 
        zorder=0)
ax.set_yscale('log')
ax.set_xlim(0, 1200)
ax.set_ylim([2e-2, 2.5])
ax.set_xlabel(r'$i$', fontsize=16)
ax.set_ylabel(r'$p_{i}$', fontsize=16)

h = []
h.append(ax.scatter(n_vector + 1, 
                    p_A_gamma_net[n_vector], 
                    marker='o', 
                    color='C0', 
                    s=70, 
                    linewidth=2, 
                    facecolor='None'))
h.append(ax.scatter(n_vector + 1, 
                    p_A_net[n_vector], 
                    marker='o', 
                    color='C2', 
                    s=70, 
                    linewidth=2, 
                    facecolor='None'))
h.append(ax.scatter(n_vector + 1, 
                    p_fully_net[n_vector], 
                    marker='x', 
                    color='k', 
                    s=30, 
                    linewidth=2))
h.append(ax.scatter(n_vector + 1, 
                    p_A_gamma2_net[n_vector], 
                    marker='^', 
                    color='C0', 
                    s=70, 
                    linewidth=2, 
                    facecolor='None'))
h.append(ax.scatter(n_vector + 1, 
                    p_A2_net[n_vector], 
                    marker='^', 
                    color='C2', 
                    s=70, 
                    linewidth=2, 
                    facecolor='None'))

ax.legend(h,[r'NB$^2$: $\rho=1$', 
             r'SL: $\rho=1$',
             r'Bayes', 
             r'NB$^2$: $\rho=2/3$', 
             r'SL: $\rho=2/3$'], 
          ncol=2, 
          fontsize=14, 
          loc='upper right', 
          bbox_to_anchor=(0.92, 1.1), 
          framealpha=1)
ax.text(30, 2.6e-2, 
        'Markers: Simulation\n Line: Theor. Bayes exponent', 
        bbox=dict(facecolor='white', alpha=0.5))

f.savefig(FIG_PATH + 'fig1_right_b.pdf', bbox_inches='tight')