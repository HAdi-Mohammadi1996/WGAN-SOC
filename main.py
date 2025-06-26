
import torch
import random
import os
import h5py
import torch.backends.cudnn as cudnn
from generator_model import Generator
from discriminator_model import Discriminator
from dataset import HDF5Dataset
from WGAN_model import Trainer
import torchvision.transforms as transforms

# Set random seed for reproducibility.
seed = 500
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Define model parameters
class ModelConfig:
    def __init__(self):
        self.dataroot = '../Sample_Subvolumes/grayscale'    # Path to input dataset
        self.out_dir_hdf5 = 'img_out_new'  # Output file for generated images  
        self.out_dir_model = 'mod_out_new' # Output file for model
        self.workers = 0    # Number of workers for data loading
        self.cuda = True    # Enable CUDA
        self.ngpu = 1     # Number of GPUs to use
        self.bsize = 512   # Batch size during training
        self.imsize = 64    # Size of training images
        self.nc = 1       # Number of channels in the input images
        self.nz = 100   # Size of the latent vector z
        self.ngf = 256   # Size of feature maps in the generator
        self.ndf = 16   # Size of feature maps in the discriminator
        self.nepochs = 200 # Number of training epochs
        self.lr = 0.00002   # Initial learning rate for optimizers
        self.beta1 = 0.1 # Beta1 hyperparameter for Adam optimizer
        self.beta2 = 0.9 # Beta2 hyperparameter for Adam optimizer
        self.save_epoch = 2 # Step for saving model checkpoints
        self.sample_interval = 50   # Interval for saving sample images

os.makedirs(str(ModelConfig().out_dir_hdf5), exist_ok=True)
os.makedirs(str(ModelConfig().out_dir_model), exist_ok=True)

# Initial learning rate
initial_lr = ModelConfig().lr

# Define learning rate schedule
def adjust_learning_rate(optimizer, epoch):
    if epoch >= 147:
        lr = 0.000001
    elif epoch >= 78:
        lr = 0.000005
    elif epoch >= 27:
        lr = 0.00001
    else:
        lr = 0.00005

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


cudnn.benchmark = True
# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available() and ModelConfig().ngpu > 0) else "cpu")
print(device, " will be used.\n")

# Get the data.
dataset = HDF5Dataset(ModelConfig().dataroot,
                          input_transform=transforms.Compose([
                          transforms.ToTensor()
                          ]))

dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=ModelConfig().bsize,
        shuffle=True, num_workers=ModelConfig().workers)

sample_batch = next(iter(dataloader))
print(sample_batch.shape)

###############################################
# Functions to be used:
###############################################
# weights initialisation
def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(w.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(w.bias.data, 0)

# # save tensor into hdf5 format
# def save_hdf5(tensor, filename):

#     tensor = tensor.cpu()
#     ndarr = tensor.mul(255).byte().numpy()
#     with h5py.File(filename, 'w') as f:
#         f.create_dataset('data', data=ndarr, dtype="i8", compression="gzip")

###############################################

# Create the generator
netG = Generator(noise_dim=ModelConfig().nz, in_channels=ModelConfig().ngf, out_channels=ModelConfig().nc, cube_size=ModelConfig().imsize//16).to(device)

if('cuda' in str(device)) and (ModelConfig().ngpu > 1):
    netG = torch.nn.DataParallel(netG, list(range(ModelConfig().ngpu)))

netG.apply(weights_init)
print(netG)

# Create the discriminator
netD = Discriminator(in_channels=ModelConfig().nc, ndf=ModelConfig().ndf, cube_size=ModelConfig().imsize//16).to(device)

if('cuda' in str(device)) and (ModelConfig().ngpu > 1):
    netD = torch.nn.DataParallel(netD, list(range(ModelConfig().ngpu)))

netD.apply(weights_init)
print(netD)

# Optimizer for the discriminator.
optimizerD = torch.optim.Adam(netD.parameters(), lr=initial_lr, betas=(ModelConfig().beta1, ModelConfig().beta2))
# Optimizer for the generator.
optimizerG = torch.optim.Adam(netG.parameters(), lr=initial_lr, betas=(ModelConfig().beta1, ModelConfig().beta2))

print("Starting Training Loop...")
print("-"*25)

# Train model
trainer = Trainer(netG, netD, optimizerG, optimizerD, ModelConfig().bsize, ModelConfig().nz,
                  gp_weight=10, critic_iterations=5, print_every=50,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, ModelConfig().nepochs, save_training_gif=True)

# Save models
name = 'SOC_WGAN_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')