import sys
sys.path.append('../')
import numpy as np
import scipy.io
import tinyDA as tda
import umbridge

### DEFINE MODEL
dataset = 'BEAM'
length = 1

# connect to the UM-Bridge Portal
umbridge_model = umbridge.HTTPModel('http://localhost:4242', 'forward')

# beam stiffness is a log-Gaussian process, so need to transform its input
E = lambda x: 1e5*np.exp(x)

# wrapping the UM-Bridge model in the tinyDA UM-Bridge interface
my_model = tda.UmBridgeModel(umbridge_model, pre=E)

# get the number of inputs and outputs
nx = umbridge_model.get_input_sizes()[0]  # input is the stiffness for each finite element along the beam
ny = umbridge_model.get_output_sizes()[0]  # outout is the deflection for each finite element

# setting up grid for input
x = np.linspace(0, length, nx)

### LOAD TRUE ###
true_data = scipy.io.loadmat('BEAM_mcmc_true.mat')
#'samples':parameters, 'yobs':d, 'xtrue':log_E_true
xtrue = true_data['xtrue'][0]
yobs = true_data['yobs'][:,1:]
true = true_data['samples']  # each row is a single sample of of logE at the 31 nodes, each column is one of the 31 nodes


print('MCMC Samps')
### LOAD MCMC ###
mcmc_samples = []  # each row will be a training set, each column will be an amt of training data
sets = 5
for mcmc_set in range(sets):
    set_samps = []
    data = scipy.io.loadmat(f'BEAM_mcmc_{mcmc_set}.mat')
    set_samps.append(data['samples'][:2500])
    set_samps.append(data['samples'][:5000])
    set_samps.append(data['samples'][:10000])
    set_samps.append(data['samples'][:25000])
    set_samps.append(data['samples'][:50000])
    set_samps.append(data['samples'][:100000])
    mcmc_samples.append(set_samps)


print('MGAN Samps')
### LOAD MGAN ###
hidden_dim = 128
normalize = 1
import torch
import scipy.io
import sys
#sys.path.append('../../')
from utilities import UnitGaussianNormalizer
device = torch.device('cpu')
params = ['2500', '5000', '10000', '25000', '50000', '100000']
sets = 5
mgan_samples = [] # each row will be a set, each column will be an amt of training data
for set_num in range(sets):
    param_samples = []

    for i in range(len(params)):
        param = params[i]

        # load data
        data = scipy.io.loadmat(f'mGAN_Training_Data/{dataset}_training_data_{param}_{set_num}.mat')
        x_train = torch.tensor(data['x_train'])
        y_train = torch.tensor(data['y_train'])

        #normalize inputs
        if normalize == 1:
            x_normalizer = UnitGaussianNormalizer(x_train)
            x_train = x_normalizer.encode(x_train)
            y_normalizer = UnitGaussianNormalizer(y_train)
            y_train = y_normalizer.encode(y_train)

        # sample prior/data
        y_in = torch.tensor(yobs).reshape((1,yobs.shape[1]))

        # normalize y
        if normalize == 1:
            y_in = y_normalizer.encode(y_in)

        # determine reference dimension
        z_dim = x_train.shape[1]
        y_dim = y_train.shape[1]

        # load network
        from learn_mgan_beam import Generator, get_noise
        gen = Generator(input_dim=z_dim+y_dim, output_dim=z_dim, hidden_dim=hidden_dim).to(device)
        folder_name = f'{param}training_mGAN/'
        file_name = f'mgan_BEAM_{param}_{set_num}_gen.pt'
        gen.load_state_dict(torch.load(folder_name+file_name))

        # sample generator
        nMC = 50000
        noise = get_noise(nMC, z_dim, device=device)
        y_in = y_in.repeat(nMC, 1)

        z = torch.cat((y_in, noise), 1).float().detach()
        x_sample = gen(z).detach()

        # transform samples back to un-normalized space
        if normalize == 1:
            x_sample = x_normalizer.decode(x_sample)

        param_samples.append(x_sample)
    mgan_samples.append(param_samples)



print('True DIV')
from sklearn.neighbors import KernelDensity
n_samples_per_dim = 1000
log_probs_true = []
all_points = []
for dim in range(31):
    true_i = true[:, dim][:, np.newaxis]
    true_kde_i = KernelDensity(bandwidth='silverman', kernel='gaussian').fit(true_i)
    points_i = true_kde_i.sample(n_samples_per_dim)
    all_points.append(points_i)
    log_true_i = true_kde_i.score_samples(points_i)
    log_probs_true.append(log_true_i)


print('MCMC KL-DIV')
mcmc_divs = [] # each row will be a set, each column will be an amt of training data
for r in range(len(mcmc_samples)):
    param_div = []
    print(f'r = {r}')
    for c in range(len(mcmc_samples[r])):
        print(f'c = {c}')
        samps = mcmc_samples[r][c]  #samples will be 31 columns and 50000 rows

        div = []
        for dim in range(31):
            samps_i = samps[:, dim][:, np.newaxis]
            samples_kde_i = KernelDensity(bandwidth="silverman", kernel='gaussian').fit(samps_i)
            log_prob_samp_i = samples_kde_i.score_samples(all_points[dim])
            vals = log_probs_true[dim] - log_prob_samp_i
            xdiv = np.mean(vals)            
            div.append(xdiv)
        div = np.sum(np.array(div))

        print(f'div = {div}')
        param_div.append(div)
    mcmc_divs.append(param_div)


print('mGAN KL-DIV')
mgan_divs = [] # each row will be a set, each column will be an amt of training data
for r in range(len(mgan_samples)):
    param_div = []
    print(f'r = {r}')
    for c in range(len(mgan_samples[r])):
        print(f'c = {c}')
        samps = mgan_samples[r][c]  #samples will be 31 columns and 50000 rows

        div = []
        for dim in range(31):
            samps_i = samps[:, dim][:, np.newaxis]
            samples_kde_i = KernelDensity(bandwidth="silverman", kernel='gaussian').fit(samps_i.numpy())
            log_prob_samp_i = samples_kde_i.score_samples(all_points[dim])
            vals = log_probs_true[dim] - log_prob_samp_i
            xdiv = np.mean(vals)            
            div.append(xdiv)
        div = np.sum(np.array(div))

        print(f'div = {div}')
        param_div.append(div)
    mgan_divs.append(param_div)


### SAVE DATA IF WANT TO REPLOT ###

data_file = 'BEAM_FinalComps_div'
import scipy.io
scipy.io.savemat(data_file + '.mat', mdict={'mcmc_divs':np.array(mcmc_divs), 'mgan_divs':np.array(mgan_divs)})


### PLOT COMPARISON ###
import matplotlib.pyplot as plt
params = ['2500', '5000', '10000', '25000', '50000', '100000']
xs = np.arange(len(params))
plt.xticks(xs, params)
for set_num in range(sets):
    plt.plot(xs, mcmc_divs[set_num], 'b.', label='MCMC' if set_num == 0 else "")
    plt.plot(xs, mgan_divs[set_num], 'r.', label='mGAN' if set_num == 0 else "")

# Calculate and plot the averages
mcmc_avg = np.mean(mcmc_divs, axis=0)
mgan_avg = np.mean(mgan_divs, axis=0)
plt.plot(xs, mcmc_avg, 'b--', label='MCMC Avg')
plt.plot(xs, mgan_avg, 'r--', label='mGAN Avg')

plt.yscale("log")
plt.legend()
plt.ylabel('log(KL-Divergence)')
plt.xlabel('Number of Model  Evaluations')
plt.savefig(f'MGAN_MCMC_FINALCOMPS.pdf')
plt.close()