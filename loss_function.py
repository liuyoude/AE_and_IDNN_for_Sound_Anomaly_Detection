import torch
import torch.nn as nn

# considering the MSE between z(Decoder1) and z_(Decoder2)
class Double_Mse_Loss(nn.Module):
    def __init__(self, lamda):
        super(Double_Mse_Loss, self).__init__()
        self.lamda = lamda
        self.criterion = nn.MSELoss()

    def forward(self, output, target, z, z_):
        res_mse = self.criterion(output, target)
        z_mse = self.criterion(z, z_)
        loss = self.lamda * res_mse + (1 - self.lamda) * z_mse
        return loss

# VAE loss function
class vae_loss(nn.Module):
    def __init__(self):
        super(vae_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
    def forward(self, recon_x, x, mu, logvar):
        mse = self.mse_loss(recon_x, x)
        kl = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu**2)
        return mse + kl


if __name__ == '__main__':
    out = torch.rand((64, 640))
    tar = torch.rand((64, 640))
    z = torch.rand((64, 128))
    z_ = torch.rand((64, 128))
    cirection = Double_Mse_Loss(0.5)
    loss = cirection(out, tar, z, z_)
    print(loss)