import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.optim as optim
from tqdm import tqdm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Creates a temporal block.
    Args:
        n_inputs (int): number of inputs.
        n_outputs (int): size of fully connected layers.
        kernel_size (int): kernel size along temporal axis of convolution layers within the temporal block.
        dilation (int): dilation of convolution layers along temporal axis within the temporal block.
        padding (int): padding
        dropout (float): dropout rate
    Returns:
        tuple of output layers
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding): #, dropout=0.2
        super(TemporalBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU()
        #self.dropout1 = nn.Dropout(dropout)

        self.conv2 = spectral_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        #self.dropout2 = nn.Dropout(dropout)
        if padding == 0:
            self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.conv2, self.chomp2, self.relu2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()
        self.init_weights()


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.5)
        self.conv2.weight.data.normal_(0, 0.5)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.5)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out,  self.relu(out + res)
    

class Generator(nn.Module):
    """Generator: 3 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self):
        super(Generator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(3, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x


class Discriminator(nn.Module):
    """Discrimnator: 1 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """ 
    def __init__(self, seq_len):
        super(Discriminator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(1, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()


class GAN:
    def __init__(self, generator, discriminator) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=6e-5, betas=(.0, 0.9), eps=1e-08)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=1.8e-4, betas=(.0, 0.9), eps=1e-08) 


    def train(self, dataloader, nz, num_epochs, clip, device, generator_path):
        t = tqdm(range(num_epochs))
        for epoch in t:
            for idx, data in enumerate(dataloader, 0):

                self.discriminator.zero_grad()
                real = data.to(device)
                noise = torch.randn(real.size(0), nz, real.size(2), device=device)
                fake = self.generator(noise).detach()
                disc_loss = -torch.mean(self.discriminator(real)) + torch.mean(self.discriminator(fake))
                disc_loss.backward()
                self.disc_optimizer.step()

                for dp in self.discriminator.parameters():
                    dp.data.clamp_(-clip, clip)
        
                if idx % 5 == 0:
                    self.generator.zero_grad()
                    gen_loss = -torch.mean(self.discriminator(self.generator(noise)))
                    gen_loss.backward()
                    self.gen_optimizer.step()
            t.set_description('Discriminator Loss: %.8f Generator Loss: %.8f' % (disc_loss.item(), gen_loss.item()))
            torch.save(self.generator, f'{generator_path}trained_generator_epoch_{epoch}.pth')
