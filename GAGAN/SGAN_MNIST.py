from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import itertools
from numpy.random import randint


# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "mnist_png/training"


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
nolayers = 5


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu, opChannels, activationG):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.activationG = activationG
        self.opChannels = opChannels

        ip = nz
        i = 0
        # Create model based on genome
        op = self.opChannels[i]

        model = []
        while i < nolayers-1:

            if i == 0:
                model += [nn.ConvTranspose2d(ip, op, 4, 1, 0, bias=False),
                          nn.BatchNorm2d(op)]
            else:
                model += [nn.ConvTranspose2d(ip, op, 4, 2, 1, bias=False),
                          nn.BatchNorm2d(op)]
            if self.activationG[i] == 0:
                model += [nn.LeakyReLU(0.2)]
            elif self.activationG[i] == 1:
                model += [nn.ReLU(inplace=True)]
            elif self.activationG[i] == 2:
                model += [nn.ELU(inplace=True)]

            ip = op
            i = i + 1

            if i < nolayers-1:
                op = self.opChannels[i]

        model += [
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()]
        self.main = torch.nn.Sequential(*model)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu,opChannels,activationF):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.activationF = activationF
        self.opChannels = opChannels


        ip = nc
        op = ndf
        i = 0
        model = []

        while i < nolayers-1:
            model += [nn.Conv2d(ip, op, 4, 2, 1, bias=False)]
            if i > 0:
                model += [nn.BatchNorm2d(op)]

            if self.activationF[i] == 0:
                model += [nn.LeakyReLU(0.2)]
            elif self.activationF[i] == 1:
                model += [nn.ReLU(inplace=True)]
            elif self.activationF[i] == 2:
                model += [nn.ELU(inplace=True)]


            ip = op
            op = self.opChannels[i]

            i = i+1
        model += [
            nn.Conv2d(ip, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()]
        self.main = torch.nn.Sequential(*model)

    def forward(self, input):
        return self.main(input)

if __name__ == '__main__':
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Grayscale(1),
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()
    # Create the generator
    activationG = [randint(0, 3) for x in range(nolayers-1)]
    # Generator output should be
    opFeatures = [randint(64, 513) for i in range(nolayers-2)]
    opFeatures.append(64)

    netG = Generator(ngpu,opFeatures,activationG).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)
    # Create the Discriminator
    activationF = [randint(0, 3) for x in range(nolayers-1)]
    opFeatures = [randint(64, 513) for i in range(nolayers-1)]
    netD = Discriminator(ngpu,opFeatures,activationF).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    criterion = torch.nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            # 20 batches per epoch- 64 bsize * 20 batches per epoch -> 1280 samples per epoch
            if i > 20:
                break

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            f_label =  torch.full((b_size,), fake_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch

            #BCE loss
            errD_real = criterion(output,label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            # label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch

            errD_fake = criterion(output,f_label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output,label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, 20,
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 20 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                #Save generator image
                path = 'C:/Users/RACHIT/Desktop/KaitavProject/LSGAN/mnist_dcgan/sgan/'
                size_figure_grid = 5
                fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
                for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                    ax[i, j].get_xaxis().set_visible(False)
                    ax[i, j].get_yaxis().set_visible(False)
                path = path + 'Epoch_{0}_'.format(epoch) + str(i) + '.png'

                for k in range(5 * 5):
                    i = k // 5
                    j = k % 5
                    ax[i, j].cla()
                    ax[i, j].imshow(fake[k, 0].cpu().data.numpy(), cmap='gray')
                    fig.savefig(path)
                plt.close()
            iters += 1


    # Get FID score for the model:
    from metrics import generative_score
    base_fid_statistics, inception_model = generative_score.initialize_fid(dataloader, sample_size=1000)
    from metrics.fid import fid_score
    from util import tools

    noise = torch.randn(1000, 100, 1, 1, device=device)
    netG.eval()
    generated_images = netG(noise).detach()
    inception_model = tools.cuda(inception_model)
    m1, s1 = fid_score.calculate_activation_statistics(
        generated_images.data.cpu().numpy(), inception_model, cuda=tools.is_cuda_available(),
        dims=2048)
    inception_model.cpu()
    m2, s2 = base_fid_statistics
    ret = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
    netG.zero_grad()
    print("Fid score is :",ret)


    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
