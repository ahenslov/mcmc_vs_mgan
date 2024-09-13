import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix

import tinyDA as tda
import umbridge

### INITIALIZING UM-BRIDGE ###
# np.random.seed(111)  # for testing purposes
dataset = 'beam'

# connect to the UM-Bridge Portal --> forward takes in stiffness (logE) and outputs deflection
umbridge_model = umbridge.HTTPModel('http://localhost:4242', 'forward')

# beam stiffness is a log-Gaussian process, so need to transform its input
E = lambda x: 1e5*np.exp(x)

# wrapping the UM-Bridge model in the tinyDA UM-Bridge interface
my_model = tda.UmBridgeModel(umbridge_model, pre=E)

### PROBLEM SETUP ###
length = 1

# get the number of inputs and outputs
nx = umbridge_model.get_input_sizes()[0]  # input is the stiffness for each finite element along the beam
ny = umbridge_model.get_output_sizes()[0]  # outout is the deflection for each finite element

# setting up grid for input
x = np.linspace(0, length, nx)

# setting up GP prior
l = 0.5  # prior length scale
C = np.exp(-0.5*distance_matrix(x[:,np.newaxis], x[:, np.newaxis])**2/l**2) # kernel matrix
my_prior = multivariate_normal(np.zeros(nx), C, allow_singular=True) # zero mean GP

# generate a draw from the prior and pass it through the model --> will use a fixed log_E_true here
log_E_true = np.array([-0.02105023,  0.11109501,  0.24320072,  0.37384151,  0.50160897,  0.62513781,
              0.743128,    0.85436262,  0.95772096,  1.0521884,   1.13686395,  1.21096465,
              1.27383042,  1.32492762,  1.36385426,  1.39034415,  1.40427288,  1.40566381,
              1.39469263,  1.37169141,  1.33714948,  1.29171238,  1.23617494,  1.17147158,
              1.09866097,  1.01890579,  0.93345006,  0.84359199,  0.7506556,   0.65596176,
              0.56079931])

d_true = my_model(log_E_true)  # true deflection

# add noise to the model output to get observations
sigma_noise = 1e-3
noise = np.random.normal(loc=0, scale=sigma_noise, size=ny)
d = d_true + noise
d[0] = 0


### PLOTTING MODEL INPUTS/OUTPUTS ###
fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(18,6))

ax[0].set_title('log(E)')
ax[0].plot(x, log_E_true, label='True')
ax[0].legend()

ax[1].set_title('Stiffness')
ax[1].plot(x, E(log_E_true), label='True')
ax[1].legend()

ax[2].set_title('Deflection')
ax[2].plot(x, d_true, label='True')
ax[2].plot(x, d, label='Observations')
ax[2].legend()
plt.savefig(dataset+'_truth_values.pdf')
plt.show()



### PLOTTING PRIOR MODEL INPUTS/OUTPUTS ###
N_pri = 50000
prior_samples = my_prior.rvs(N_pri)
n_plotted = 500
pri_ids = np.random.randint(0, N_pri, n_plotted)

fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(18,6))

ax[0].set_title('log(E)')
for i in pri_ids:
    ax[0].plot(x, prior_samples[i], c='k', alpha=0.02)
ax[0].plot(x, log_E_true, label='True')
ax[0].legend()

ax[1].set_title('Stiffness')
for i in pri_ids:
    ax[1].plot(x, E(prior_samples[i]), c='k', alpha=0.02)
ax[1].plot(x, E(log_E_true), label='True')
ax[1].legend()

ax[2].set_title('Deflection')
for i in pri_ids:
    ax[2].plot(x, my_model(prior_samples[i]), c='k', alpha=0.02)
ax[2].plot(x, d_true, label='True')
ax[2].legend()
plt.savefig(dataset+'_prior_draws.pdf')
plt.show()


### ESTABLISHING THE LIKELIHOOD ###
sigma_likelihood = 1e-2
cov_likelihood = sigma_likelihood**2*np.eye(ny)
my_loglike = tda.GaussianLogLike(d, cov_likelihood)

### INITIALIZE THE POSTERIOR ###
my_posterior = tda.Posterior(my_prior, my_loglike, my_model)


### SETTING UP PROPOSAL ###
# using an adaptive preconditioned Crank-Nicolson b/c prior is of type scipy.stats.multivariate_normal
pcn_scaling = 0.1
pcn_adaptive = True
my_proposal = tda.CrankNicolson(scaling=pcn_scaling, adaptive=pcn_adaptive)


### SAMPLING ###
iterations = 520000
burnin = 20000
my_chains = tda.sample(my_posterior, my_proposal, iterations=iterations, n_chains=2, force_sequential=True)
#Samples generated are possible values for log(E) 


### GET DATA ###
# convert to ArViz InferenceData object
import arviz as az
data = tda.to_inference_data(my_chains, burnin=burnin)

# display posterior summary stats
from IPython.display import display
summary_stats = az.summary(data)
display(summary_stats)

# plot posterior kernel densities and traces
for i in range(31):
    axes = az.plot_trace(data, var_names=[f'x{i}'])
    axes[0,0].axvline(log_E_true[i], color='r', linewidth=3)
    axes[0,1].axhline(log_E_true[i], color='r', linewidth=3)
    plt.savefig(f'{dataset}_posterior_trace_plot_{i}.pdf')
    #plt.show()
    plt.close()

# extract the parameters from the chains  --> parameters is the logE samples
parameters = [link.parameters for link in my_chains['chain_0'][burnin+1:] + my_chains['chain_1'][burnin+1:]]


# plot some posterior draws of the model input and output.
n_samples = 1000
ids = np.random.randint(0, len(parameters), n_samples)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
fig.suptitle('True Posterior Draws')
ax[0].set_title('log(E)')
for i in ids:
    ax[0].plot(x, parameters[i], c='k', alpha=0.01)
ax[0].set_ylim(-3, 3)
ax[0].plot(x, log_E_true, label='True')
ax[0].legend()
ax[1].set_title('Stiffness')
for i in ids:
    ax[1].plot(x, E(parameters[i]), c='k', alpha=0.01)
ax[1].set_ylim(0, 1000000)
ax[1].plot(x, E(log_E_true), label='True')
ax[1].legend()
ax[2].set_title('Deflection')
for i in ids:
    ax[2].plot(x, my_model(parameters[i]), c='k', alpha=0.01)
ax[2].set_ylim(0, .14)
ax[2].plot(x, d_true, label='True')
ax[2].legend()
plt.savefig(dataset+'_mcmc_posterior_draws.pdf')
plt.show()


# compute acceptance rate
set0 = set(my_chains['chain_0'])
set1 = set(my_chains['chain_1'])
acc0 = len(set0) / len(my_chains["chain_0"])
acc1 = len(set1) / len(my_chains["chain_1"])
print(f'Acceptance rate in chain0  =  {acc0}')
print(f'Acceptance rate in chain1  =  {acc1}')


# save results
data_file = 'BEAM_mcmc'
import scipy.io
scipy.io.savemat(data_file + '.mat', mdict={'samples':parameters, 'yobs':d, 'xtrue':log_E_true, 'acceptance_rates':np.array([acc0,acc1])})
