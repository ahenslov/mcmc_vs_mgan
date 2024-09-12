import sys
sys.path.append('../')
import numpy as np
import scipy.io
import tinyDA as tda
import umbridge
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
import torch


set_num = 0


### DEFINE MODEL ###
dataset = 'BEAM'
length = 1
umbridge_model = umbridge.HTTPModel('http://localhost:4242', 'forward')
E = lambda x: 1e5*np.exp(x)
my_model = tda.UmBridgeModel(umbridge_model, pre=E)
nx = umbridge_model.get_input_sizes()[0]  # input is the stiffness for each finite element along the beam
ny = umbridge_model.get_output_sizes()[0]  # outout is the deflection for each finite element
x = np.linspace(0, length, nx)



### GATHER SET OF OBS TO GENERATE POSTERIOR OF ###
len_training = 100000

joint_dist = scipy.io.loadmat(f'mGAN_Training_Data/BEAM_training_data_{len_training}_{set_num}.mat')
y_dist = joint_dist['y_train']

N_obs = 100
obs_idx = np.random.randint(0, len_training, N_obs)
random_ys = [y_dist[i] for i in obs_idx]  # 100 rows, 30 cols each

print('Random Observations Gathered')



### GENERATE POSTERIORS FOR DIFFERENT OBS ###
hidden_dim = 128
normalize = 1
import torch
import scipy.io
import sys
#sys.path.append('../../')
from utilities import UnitGaussianNormalizer
device = torch.device('cpu')

# load data
data = scipy.io.loadmat(f'mGAN_Training_Data/BEAM_training_data_{len_training}_{set_num}.mat')
x_train = torch.tensor(data['x_train'])
y_train = torch.tensor(data['y_train'])

#normalize inputs
if normalize == 1:
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

# determine reference dimension
z_dim = x_train.shape[1]
y_dim = y_train.shape[1]

# load network
from learn_mgan_beam import Generator, get_noise
gen = Generator(input_dim=z_dim+y_dim, output_dim=z_dim, hidden_dim=hidden_dim).to(device)
folder_name = f'{len_training}training_mGAN/'
file_name = f'mgan_BEAM_{len_training}_{set_num}_gen.pt'
gen.load_state_dict(torch.load(folder_name+file_name))

posterior_samps = []

for yobs in random_ys:
    yobs = np.array([yobs])

    # sample prior/data
    y_in = torch.tensor(yobs).reshape((1,yobs.shape[1]))

    # normalize y
    if normalize == 1:
        y_in = y_normalizer.encode(y_in)

    # sample generator
    nMC = 50000
    noise = get_noise(nMC, z_dim, device=device)
    y_in = y_in.repeat(nMC, 1)

    z = torch.cat((y_in, noise), 1).float().detach()
    x_sample = gen(z).detach()

    # transform samples back to un-normalized space
    if normalize == 1:
        x_sample = x_normalizer.decode(x_sample)

    posterior_samps.append(x_sample)

print('Posterior Samples Gathered')



### DEFINE PRIOR AND SAMPLE ###
l = 0.5  # prior length scale
C = np.exp(-0.5*distance_matrix(x[:,np.newaxis], x[:, np.newaxis])**2/l**2) # kernel matrix
my_prior = multivariate_normal(np.zeros(nx), C, allow_singular=True) # zero mean GP

N_pri = 100000
prior_samples = my_prior.rvs(N_pri)  ##definitely better way to do it?? --> split in each dim and use mtv_norm.logpdf


### COMPUTE KL-DIV ###
from sklearn.neighbors import KernelDensity
n_samples_per_dim = 1000

def compute_kl_div(posterior, prior):
    """Calculates the kl-divergence in a singular dimension (still 2d array of inputs) for a singular posterior"""
    post_kde = KernelDensity(bandwidth="silverman", kernel='gaussian').fit(posterior)
    pri_kde = KernelDensity(bandwidth="silverman", kernel='gaussian').fit(prior) #split and use .logpdf
    x_points = post_kde.sample(n_samples_per_dim)
    log_post = post_kde.score_samples(x_points)
    log_pri = pri_kde.score_samples(x_points)
    vals = log_post - log_pri
    xdiv = np.mean(vals)
    return xdiv

divs = []  # each row will be a different posterior/yobs, and each column will be a different dimension
for posterior_samps_i in posterior_samps:
    yi_divs = []  #should be of 31 dims, each value represents a diff dim
    for dim in range(31):
        post_samps = posterior_samps_i[:, dim][:, np.newaxis]
        pri_samps = prior_samples[:, dim][:, np.newaxis]
        div_i = compute_kl_div(post_samps, pri_samps)
        yi_divs.append(div_i)
    divs.append(yi_divs)

divs = np.array(divs) 



### COMPUTE MEANS AND STD_ERROR ###
means = np.mean(divs, axis=0) #want to have mean across each column, so means is of shape 31
std_error = 1.96/np.sqrt(N_obs) * np.std(divs, axis=0) #want to have mean across each column, so std is of shape 31

### SAVE RESULTS ###
scipy.io.savemat('EIG_calculations.mat', mdict={'divs':divs, 'means':means, 'std_error':std_error})

### PLOT COMPARISON ###
dims = np.arange(31)
xs = dims + 1
plt.figure(figsize=(10, 6))
plt.xticks(xs[0::5], dims[0::5])
plt.plot(xs, means, 'r:', linewidth=1)
plt.errorbar(xs, means, yerr=std_error, capsize=2, fmt="o", color='k', markersize=2, linewidth=1)
plt.xlabel('Nth Dimension of X')
plt.ylabel('KL-Divergence')
plt.ylim(bottom=0)
plt.title('EIG Calculations using mGAN')
plt.savefig(f'EIG_Calculations.pdf')
plt.close()
