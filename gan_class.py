import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from numpy.random import randint
import itertools
from util.ConvLayer import Conv2dSame
from metrics.fid import fid_score
from util import tools
from metrics import generative_score


###############################################################################################################################
# ########################################################## Network Descriptor #######################################################################################################################################################################################


class NetworkDescriptor:
    def __init__(self, number_layers=1, input_dim=1, output_dim=1,  list_ouput_channels=None, init_functions=1, list_act_functions=None, number_loop_train=1,lossfunction=0):
        self.number_layers = number_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.list_ouput_channels = list_ouput_channels
        self.number_loop_train = number_loop_train
        self.init_functions = init_functions
        self.list_act_functions = list_act_functions
        self.ngf = output_dim
        self.ndf = output_dim
        self.nc = input_dim
        self.lossfunction = lossfunction

        #constants
        self.nz = 100
        self.same_conv_pos = {5: [0, 0, 0, 0, 0],
                     6: [0, 1, 0, 0, 0, 0],
                     7: [0, 1, 0, 1, 0, 0, 0],
                     8: [0, 1, 0, 1, 0, 1, 0, 0],
                     9: [0, 1, 0, 1, 0, 1, 0, 1, 0],
                     10: [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]}

    def copy_from_other_network(self, other_network):
        self.number_layers = other_network.number_layers
        self.input_dim = other_network.input_dim
        self.number_loop_train = other_network.number_loop_train
        self.init_functions = other_network.init_functions
        self.output_dim = other_network.output_dim
        self.list_ouput_channels = copy.deepcopy(other_network.list_ouput_channels)
        self.list_act_functions = copy.deepcopy(other_network.list_act_functions)
        self.nz = 100


###############################################################################################################################
# ########################################################## GAN Descriptor  #######################################################################################################################################################################################


class GANDescriptor:
    def __init__(self, input_dim=1, output_dim=64, lossfunction=0, lrate=0.0001,dataloader=None,epochs=20):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lossfunction = lossfunction
        self.Gen_network = None
        self.Disc_network = None
        self.lrate = lrate
        self.dataloader = dataloader
        self.epochs = epochs
        # TO recreate the model incase of specific mutation types

    def copy_from_other(self, other):
        self.input_dim = other.input_dim
        self.output_dim = other.output_dim
        self.lossfunction = other.lossfunction
        self.Gen_network = copy.deepcopy(other.Gen_network)     # These are  Network_Descriptor structures
        self.Disc_network = copy.deepcopy(other.Disc_network)
        self.lrate = copy.deepcopy(other.lrate)


    def gan_generator_initialization(self, generator_layers=1, generator_input_dim=None, generator_output_dim=None,
                                     generator_list_ouput_channels=None, generator_init_functions=1, generator_list_act_functions=None,generator_number_loop_train=1):

        self.Gen_network = NetworkDescriptor(generator_layers, generator_input_dim, generator_output_dim, generator_list_ouput_channels, generator_init_functions,
                                             generator_list_act_functions, generator_number_loop_train,self.lossfunction)

    def gan_discriminator_initialization(self, discriminator_layers = 1, discriminator_input_dim = None,discriminator_output_dim=None,
                                         discriminator_list_ouput_channels=None, discriminator_init_functions=1, discriminator_list_act_functions=None,discriminator_number_loop_train=1):
        self.Disc_network = NetworkDescriptor(discriminator_layers, discriminator_input_dim,
                                              discriminator_output_dim, discriminator_list_ouput_channels, discriminator_init_functions,
                                              discriminator_list_act_functions, discriminator_number_loop_train,self.lossfunction)


###############################################################################################################################
# ########################################################## Network #######################################################################################################################################################################################



class Generator(nn.Module):
    def __init__(self,  g_descriptor):
        super(Generator, self).__init__()
        self.activationG = g_descriptor.list_act_functions
        self.nolayers = g_descriptor.number_layers
        self.opChannels = g_descriptor.list_ouput_channels
        self.same_conv_pos = g_descriptor.same_conv_pos


        ip = g_descriptor.nz
        i = 0
        op = self.opChannels[i]

        model = []

        # Create model based on genome
        while i < (self.nolayers-1):

            # Set convtraspose and batch norm layers
            if i == 0:
                model += [nn.ConvTranspose2d(ip, op, 4, 1, 0, bias=False),
                          nn.BatchNorm2d(op)]
            elif (self.same_conv_pos[self.nolayers][i]):
                model += [nn.ConvTranspose2d(ip, op, 3, 1, 1, bias=False),
                          nn.BatchNorm2d(op)]
            else:
                model += [nn.ConvTranspose2d(ip, op, 4, 2, 1, bias=False),
                          nn.BatchNorm2d(op)]

            # Followed by activation functions
            if self.activationG[i] == 0:
                model += [nn.LeakyReLU(0.2)]
            elif self.activationG[i] == 1:
                model += [nn.ReLU(inplace=True)]
            elif self.activationG[i] == 2:
                model += [nn.ELU(inplace=True)]

            ip = op
            i = i + 1

            # Preventing indexout of errpr for output channels
            if i < (self.nolayers-1):
                op = self.opChannels[i]

        # The last layer remains the same
        model += [
            nn.ConvTranspose2d(g_descriptor.ngf, g_descriptor.nc, 4, 2, 1, bias=False),
            nn.Tanh()]
        self.main = torch.nn.Sequential(*model)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, d_descriptor):
        super(Discriminator, self).__init__()
        self.activationD = d_descriptor.list_act_functions

        self.nolayers = d_descriptor.number_layers
        self.opChannels = d_descriptor.list_ouput_channels
        self.same_conv2d_pos = d_descriptor.same_conv_pos
        ip = d_descriptor.nc
        op = d_descriptor.ndf
        i = 0
        model = []

        while i < (self.nolayers-1):

            if self.same_conv2d_pos[self.nolayers][i] == 1:
                # Add conv2d_same layers !
                model += [Conv2dSame(ip, op, (4, 4), 1)]
            else:
                model += [nn.Conv2d(ip, op, 4, 2, 1)]

            # Normalization is only added if the layer is not the input layer
            if i > 0:
                model += [nn.BatchNorm2d(op)]

            if self.activationD[i] == 0:
                model += [nn.LeakyReLU(0.2)]
            elif self.activationD[i] == 1:
                model += [nn.ReLU(inplace=True)]
            elif self.activationD[i] == 2:
                model += [nn.ELU(inplace=True)]

            ip = op
            i = i+1
            if i < self.nolayers:
                op = self.opChannels[i]


        model += [nn.Conv2d(ip, 1, 4, 1, 0, bias=False)]

        # If the loss function is BCE then only add the sigmoid function!
        if d_descriptor.lossfunction == 0:
            model += [nn.Sigmoid()]
        self.main = torch.nn.Sequential(*model)

    def forward(self, input):
        return self.main(input)

###################################################################################################
# ############################################ GAN  ###############################################
###################################################################################################


class GAN:

    def __init__(self, gan_descriptor):
        # Decide which device we want to run on
        ngpu = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.descriptor = gan_descriptor
        self.loss_function = self.descriptor.lossfunction
        self.Disc_network = Discriminator(self.descriptor.Disc_network).to(self.device)
        self.init_d_model = True
        self.Gen_network = Generator(self.descriptor.Gen_network).to(self.device)
        self.init_g_model = True
        self.dataloader =self.descriptor.dataloader
        # DL is now with GAN class no need to keep it with descriptor- can also make one common data loader
        self.descriptor.dataloader = None
        self.epochs = self.descriptor.epochs

    def reset_network(self):
        self.Gen_network.reset_network()
        self.Disc_network.reset_network()

    # custom weights initialization called on netG and netD
    def weights_init(self,m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)



    def getFIDScore(self):


        #INIT INCEPTION MODEL ONLY ONCE(!)

        # Get FID score for the model:
        base_fid_statistics, inception_model = generative_score.initialize_fid(self.dataloader, sample_size=1000)

        noise = torch.randn(1000, 100, 1, 1, device=self.device)

        self.Gen_network.eval()
        generated_images = self.Gen_network(noise).detach()
        inception_model = tools.cuda(inception_model)
        m1, s1 = fid_score.calculate_activation_statistics(
            generated_images.data.cpu().numpy(), inception_model, cuda=tools.is_cuda_available(),
            dims=2048)
        inception_model.cpu()
        m2, s2 = base_fid_statistics
        ret = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
        self.Gen_network.zero_grad()
        return ret

    def train_gan(self):

        ##########
        # Constants for training
        ############
        nz = 100
        # Beta1 hyperparam for Adam optimizers
        beta1 = 0.5
        # Set random seem for reproducibility
        manualSeed = 999
        # manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        if(self.init_g_model):
            self.Gen_network.apply(self.weights_init)
            self.optimizerG = optim.Adam(self.Gen_network.parameters(), lr=self.descriptor.lrate, betas=(beta1, 0.999))
            self.init_g_model = False
        if(self.init_d_model):
            self.Disc_network.apply(self.weights_init)
            self.optimizerD = optim.Adam(self.Disc_network.parameters(), lr=self.descriptor.lrate, betas=(beta1, 0.999))
            self.init_d_model = False

        # ADD DIFFERENT LOSS FUCNTIONS

        if self.loss_function ==0:
            criterion = torch.nn.BCELoss()
        elif (self.loss_function == 2 or 3):
            BCE_stable = torch.nn.BCEWithLogitsLoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)
        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        print("Starting Training Loop...")
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        # For each epoch
        for epoch in range(self.epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                # 20 batches per epoch- 64 bsize * 20 batches per epoch -> 1280 samples per epoch
                if i > self.epochs:
                    break

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.Disc_network.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)

                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=self.device)
                f_label = torch.full((b_size,), fake_label, device=self.device)
                # Forward pass real batch through D
                y_pred = self.Disc_network(real_cpu).view(-1)


                # Calculate loss on all-real batch
                if self.loss_function == 0:
                    # BCE loss
                    errD_real = criterion(y_pred, label)
                    errD_real.backward()
                elif self.loss_function ==1:
                    # MSE loss
                    errD_real = torch.mean((y_pred - label) ** 2)
                    errD_real.backward()
                elif self.loss_function == 4:
                    #Hinge loss
                    errD_real = torch.mean(torch.nn.ReLU()(1.0 - y_pred))
                    errD_real.backward()
                D_x = y_pred.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.Gen_network(noise)
                # label.fill_(fake_label)
                # Classify all fake batch with D
                y_pred_fake = self.Disc_network(fake.detach()).view(-1)

                if self.loss_function == 0:
                    # BCE loss
                    errD_fake = criterion(y_pred_fake, f_label)
                    errD_fake.backward()
                    errD = errD_real + errD_fake
                elif self.loss_function == 1:
                    # MSE loss
                    errD_fake = torch.mean((y_pred_fake)**2)
                    errD_fake.backward()
                    errD = errD_real + errD_fake
                elif self.loss_function ==2:
                    # RSGAN
                    label = torch.full((b_size,), real_label, device=self.device)
                    errD = BCE_stable(y_pred - y_pred_fake, label)
                    errD.backward()
                elif self.loss_function == 3:
                    #RAGAN
                    # Add the gradients from the all-real and all-fake batches
                    y = torch.full((b_size,), real_label, device=self.device)
                    y2 = torch.full((b_size,), fake_label, device=self.device)
                    errD = (BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(
                        y_pred_fake - torch.mean(y_pred),y2)) / 2
                    errD.backward()
                elif self.loss_function == 4:
                    errD = torch.mean(torch.nn.ReLU()(1.0 + y_pred_fake))
                    # Calculate the gradients for this batch
                    errD.backward()


                # Calculate the gradients for this batch

                D_G_z1 = y_pred_fake.mean().item()
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.Gen_network.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                y_pred_fake = self.Disc_network(fake).view(-1)
                # Calculate G's loss based on this output
                if self.loss_function == 0:
                    #BCE
                    errG = criterion(y_pred_fake, label)
                elif self.loss_function ==1 :
                    #MSE
                    errG = torch.mean((y_pred_fake - label) ** 2)
                elif self.loss_function == 2:
                    #RSGAN
                    label = torch.full((b_size,), real_label, device=self.device)
                    y_pred = self.Disc_network(real_cpu).view(-1)
                    errG = BCE_stable(y_pred_fake - y_pred, label)
                elif self.loss_function == 3:
                    #RAGAN
                    y_pred = self.Disc_network(real_cpu).view(-1)
                    y = torch.full((b_size,), real_label, device=self.device)
                    y2 = torch.full((b_size,), fake_label, device=self.device)
                    errG = (BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(
                        y_pred_fake - torch.mean(y_pred),y)) / 2
                elif self.loss_function == 4:
                    #Higen loss
                    errG = -torch.mean(y_pred_fake)


                # Calculate gradients for G
                errG.backward()
                D_G_z2 = y_pred_fake.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 10 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.epochs, i, 20,
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 20 == 0) or ((epoch == self.epochs- 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.Gen_network(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                    # THINK ABOUT SAVING THE RESULTS AND IMAGES AND MODEL
                    path = 'C:/Users/RACHIT/Desktop/GAGAN/output_images/'
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





###################################################################################################
# ############################################ TEST MAIN FUNCTION  ###############################################
###################################################################################################
# def main():
#
#
#     # Root directory for dataset
#     dataroot = "mnist_png/training"
#     image_size = 64
#     dataset = dset.ImageFolder(root=dataroot,
#                                transform=transforms.Compose([
#                                    transforms.Grayscale(1),
#                                    transforms.Resize(image_size),
#                                    transforms.CenterCrop(image_size),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                ]))
#     # Create the dataloader
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
#                                              shuffle=True, num_workers=2)
#
#
#     input_channel =1
#     output_dim =64
#     lrate = 0.0002
#     lossfunction = 2
#     epochs = 20
#     my_gan_descriptor = GANDescriptor(input_channel, output_dim, lossfunction, lrate,dataloader,epochs)
#
#     # g_layer = np.random.randint(2,11) # Number of hidden layers
#     g_layer = 5
#     g_activations = [randint(0, 3) for x in range(g_layer - 1)]
#
#     gchannels={5: [512,256,128,64,64],
#               6: [512,512,256,128,64,64],
#               7: [512,512,256,256,128,64,64],
#               8: [512,512,256,256,128,128,64,64],
#               9: [512,512,256,256,128,128,64,64,64],
#               10:[512,512,512,256,256,128,128,64,64,64]}
#     g_opChannels = gchannels[g_layer]
#
#
#     g_weight_init = 0
#     g_loop = 1
#     nz = 100
#
#     my_gan_descriptor.gan_generator_initialization(g_layer, input_channel,output_dim,g_opChannels,
#                                                        g_weight_init,g_activations,g_loop)
#
#     # d_layer = np.random.randint(2,9) # Number of hidden layers
#     d_layer = 5
#     d_weight_init = 0
#     d_activations = [randint(0, 3) for x in range(d_layer - 1)]
#     dchannels={5: [64,128,256,512,512],
#               6: [64,64,128,256,512,512],
#               7: [64,64,128,128,256,512,512],
#               8: [64,64,128,128,256,256,512,512],
#               9: [64,64,128,128,256,256,512,512,512],
#               10:[64,64,64,128,128,256,256,512,512,512]}
#     d_opChannels = dchannels[d_layer]
#     d_loop = 1
#     my_gan_descriptor.gan_discriminator_initialization(d_layer, input_channel,output_dim,d_opChannels,
#                                                    d_weight_init,d_activations ,d_loop)
#
#
#     individual = GAN(my_gan_descriptor)
#     print(individual.Gen_network)
#     print(individual.Disc_network)
#
#     individual.train_gan()
# if __name__ == '__main__':
#     # freeze_support() here if program needs to be frozen
#     main()  # exec