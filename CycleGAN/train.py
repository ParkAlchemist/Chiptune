import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Generator, Discriminator

# Initialize generators and discriminators
G = Generator()
F = Generator()
D_X = Discriminator()
D_Y = Discriminator()

# Define loss functions
cycle_loss = nn.L1Loss()
adversarial_loss = nn.MSELoss()

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_F = torch.optim.Adam(F.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        real_X, real_Y = batch

        # Train generators
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        fake_Y = G(real_X)
        fake_X = F(real_Y)
        cycle_X = F(fake_Y)
        cycle_Y = G(fake_X)
        loss_cycle_X = cycle_loss(cycle_X, real_X)
        loss_cycle_Y = cycle_loss(cycle_Y, real_Y)
        loss_G = adversarial_loss(D_Y(fake_Y), torch.ones_like(D_Y(fake_Y)))
        loss_F = adversarial_loss(D_X(fake_X), torch.ones_like(D_X(fake_X)))
        total_loss_G = loss_G + loss_cycle_X + loss_cycle_Y
        total_loss_F = loss_F + loss_cycle_X + loss_cycle_Y
        total_loss_G.backward()
        total_loss_F.backward()
        optimizer_G.step()
        optimizer_F.step()

        # Train discriminators
        optimizer_D_X.zero_grad()
        optimizer_D_Y.zero_grad()
        loss_D_X = adversarial_loss(D_X(real_X), torch.ones_like(D_X(real_X))) + adversarial_loss(D_X(fake_X.detach()), torch.zeros_like(D_X(fake_X)))
        loss_D_Y = adversarial_loss(D_Y(real_Y), torch.ones_like(D_Y(real_Y))) + adversarial_loss(D_Y(fake_Y.detach()), torch.zeros_like(D_Y(fake_Y)))
        loss_D_X.backward()
        loss_D_Y.backward()
        optimizer_D_X.step()
        optimizer_D_Y.step()
