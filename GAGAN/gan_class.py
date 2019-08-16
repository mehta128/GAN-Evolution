import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
###################################################################################################
# ######## Auxiliary Functions
###################################################################################################


# def plot(samples, theshape):
#     fig = plt.figure(figsize=(5, 5))
#     gs = gridspec.GridSpec(5, 5)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, sample in enumerate(samples[:25, :]):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.reshape(theshape))
#
#     return fig
#
#
# def next_random_batch(num, data):
#     """
#     Return a total of `num` random samples and labels.
#     """
#     idx = np.arange(0, len(data))
#     np.random.shuffle(idx)
#     idx = idx[:num]
#     data_shuffle = data[idx, :]
#     return data_shuffle
#
#
# def next_batch(num, data, start):
#     """
#     Return a total of 'num' samples and labels.
#     """
#     idx = np.arange(start, np.min([start+num, len(data)]))
#     return data[idx, :]
#
#
# def xavier_init(shape):
#     in_dim = shape[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#
#     return tf.random_normal(shape=shape, stddev=tf.cast(xavier_stddev, "float32"))

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
        self.nz = 100
        self.ngf = output_dim
        self.ndf = output_dim
        self.nc = input_dim
        self.lossfunction = lossfunction

    def copy_from_other_network(self, other_network):
        self.number_layers = other_network.number_layers
        self.input_dim = other_network.input_dim
        self.number_loop_train = other_network.number_loop_train
        self.init_functions = other_network.init_functions
        self.output_dim = other_network.output_dim
        self.list_ouput_channels = copy.deepcopy(other_network.list_ouput_channels)
        self.list_act_functions = copy.deepcopy(other_network.list_act_functions)
        self.nz = 100

    def network_add_layer(self, layer_pos, output_dims, init_a_function):
        """
        Function: network_add_layer()
        Adds a layer at a specified position, with a given  number of output channel, activation function.
        If the layer is inserted in layer_pos \in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos >= self.number_layers:
            return

        # We create the new layer and add it to the network descriptor
        self.list_ouput_channels.insert(layer_pos, output_dims)
        self.list_act_functions.insert(layer_pos, init_a_function)

        # Finally the number of hidden layers is updated
        self.number_layers = self.number_layers + 1

    def network_remove_layer(self, layer_pos):
        """
        Function: network_remove_layer()
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos \in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        # If not within feasible bounds, return
        if layer_pos <= 1 or layer_pos > self.number_layers:
            return

        # We set the number of input and output dimensions for the layer to be
        # added and for the ones in the architecture that will be connected to it

        # We delete the layer in pos layer_pos
        self.list_ouput_channels.pop(layer_pos)
        self.list_act_functions.pop(layer_pos)

        # Finally the number of hidden layers is updated
        self.number_layers = self.number_layers - 1

    def network_remove_random_layer(self):
        layer_pos = np.random.randint(self.number_layers)
        self.network_remove_layer(layer_pos)

    def change_activation_fn_in_layer(self, layer_pos, new_act_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_layers:
            return
        self.list_act_functions[layer_pos] = new_act_fn

    def change_dimensions_in_layer(self, layer_pos, new_dim):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_layers:
            return
        # If the dimension of the layer is identical to the existing one, return
        self.list_ouput_channels[layer_pos] = new_dim

    def change_dimensions_in_random_layer(self, max_layer_size):
        layer_pos = np.random.randint(self.number_layers)
        new_dim = np.random.randint(max_layer_size)+1
        self.change_dimensions_in_layer(layer_pos, new_dim)


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

        ip = g_descriptor.nz
        i = 0
        op = self.opChannels[i]

        model = []

        # Create model based on genome
        while i < (self.nolayers-1):

            if i == 0:
                model += [nn.ConvTranspose2d(ip, op, 4, 1, 0, bias=False),
                          nn.BatchNorm2d(op)]
            else:

                # Add two strides if layers is not the first one
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

        ip = d_descriptor.nc
        op = d_descriptor.ndf
        i = 0
        model = []

        while i < (self.nolayers-1):
            model += [nn.Conv2d(ip, op, 4, 2, 1, bias=False)]

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
            op = self.opChannels[i]

            i = i+1

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
        self.Gen_network = Generator(self.descriptor.Gen_network).to(self.device)
        self.dataloader =self.descriptor.dataloader
        self.epochs = self.descriptor.epochs
    def reset_network(self):
        self.Gen_network.reset_network()
        self.Disc_network.reset_network()

    # custom weights initialization called on netG and netD
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

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


        #######################
        # ADD DIFFERENT WEIGHTS INIT
        self.Gen_network.apply(self.weights_init)
        self.Disc_network.apply(self.weights_init)


        # ADD DIFFERENT LOSS FUCNTIONS
        criterion = torch.nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)
        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0
        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(self.Disc_network.parameters(), lr= self.descriptor.lrate, betas=(beta1, 0.999))
        optimizerG = optim.Adam(self.Gen_network.parameters(), lr= self.descriptor.lrate, betas=(beta1, 0.999))

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

                print(real_cpu.size())

                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=self.device)
                f_label = torch.full((b_size,), fake_label, device=self.device)
                # Forward pass real batch through D
                output = self.Disc_network(real_cpu).view(-1)
                # Calculate loss on all-real batch

                print(output.size())
                print(label.size())

                # BCE loss
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.Gen_network(noise)
                # label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.Disc_network(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch

                errD_fake = criterion(output, f_label)
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
                self.Gen_network.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.Disc_network(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

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
                    path = 'C:/Users/RACHIT/Desktop/GAN_Evolution-master/output_images/'
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
def main():


    # Root directory for dataset
    dataroot = "mnist_png/training"
    image_size = 64
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
                                             shuffle=True, num_workers=2)


    input_channel =1
    output_dim =64
    lrate = 0.0002
    lossfunction = 0
    epochs = 20
    my_gan_descriptor = GANDescriptor(input_channel, output_dim, lossfunction, lrate,dataloader,epochs)

    g_layer = np.random.randint(2,11) # Number of hidden layers
    g_layer = 5
    g_activations = [randint(0, 3) for x in range(g_layer - 1)]
    g_opChannels = [randint(64, 513) for i in range(g_layer - 2)]
    # Generator output should be
    g_opChannels.append(64)
    g_weight_init = 0
    g_loop = 1
    nz = 100

    my_gan_descriptor.gan_generator_initialization(g_layer, input_channel,output_dim,g_opChannels,
                                                       g_weight_init,g_activations,g_loop)

    d_layer = np.random.randint(2,9) # Number of hidden layers
    d_layer = 6
    d_weight_init = 0
    d_activations = [randint(0, 3) for x in range(d_layer - 1)]
    d_opChannels = [randint(64, 513) for i in range(d_layer - 1)]
    d_loop = 1
    my_gan_descriptor.gan_discriminator_initialization(d_layer, input_channel,output_dim,d_opChannels,
                                                   d_weight_init,d_activations ,d_loop)


    individual = GAN(my_gan_descriptor)
    print(individual.Gen_network)
    print(individual.Disc_network)

    individual.train_gan()
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # exec