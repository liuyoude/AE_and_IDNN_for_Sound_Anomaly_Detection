import torch
import torch.nn as nn
from torchsummary import summary
#from tensorboardX import SummaryWriter

# linear block
class Liner_Module(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Liner_Module, self).__init__()
        self.liner = nn.Linear(input_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        x = self.liner(input)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Auto_encoder(nn.Module):
    def __init__(self, input_dim=640, output_dim=640):
        super(Auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            Liner_Module(input_dim=input_dim, out_dim=256),
            Liner_Module(input_dim=256, out_dim=128),
            Liner_Module(input_dim=128, out_dim=64),
            Liner_Module(input_dim=64, out_dim=32),
            Liner_Module(input_dim=32, out_dim=16),
        )
        self.decoder = nn.Sequential(
            Liner_Module(input_dim=16, out_dim=32),
            Liner_Module(input_dim=32, out_dim=64),
            Liner_Module(input_dim=64, out_dim=128),
            Liner_Module(input_dim=128, out_dim=256),
            nn.Linear(256, output_dim),
        )

    def forward(self, input: torch.Tensor):
        x_feature = self.encoder(input)
        x = self.decoder(x_feature)
        return x, x_feature

class VAE(nn.Module):
    def __init__(self, input_dim=640, output_dim=640):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            Liner_Module(input_dim=input_dim, out_dim=512),
            Liner_Module(input_dim=512, out_dim=256),
            Liner_Module(input_dim=256, out_dim=128),
            Liner_Module(input_dim=128, out_dim=64),
        )
        self.fc_mean = Liner_Module(input_dim=64, out_dim=16)
        self.fc_logvar = Liner_Module(input_dim=64, out_dim=16)

        self.decoder = nn.Sequential(
            Liner_Module(input_dim=16, out_dim=32),
            Liner_Module(input_dim=32, out_dim=64),
            Liner_Module(input_dim=64, out_dim=128),
            Liner_Module(input_dim=128, out_dim=256),
            nn.Linear(256, output_dim),
        )

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        rand = torch.randn(std.size())
        if mu.is_cuda:
            rand = rand.cuda()
        z = rand * std + mu
        return z

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterization(mu, logvar)
        output = self.decoder(z)
        return output, mu, logvar

class LSTM_AE(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super(LSTM_AE, self).__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )

        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=256,
            num_layers=2,
            dropout=0,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(256, input_dim)

    def forward(self, x):
        if not hasattr(self, '_flattened'):
            self.encoder.flatten_parameters()
            self.decoder.flatten_parameters()
        batch, frames, dim = x.size()
        output, en_hidden = self.encoder(x)
        reconstruct_output = []
        temp_input = torch.zeros((batch, 1, 128), dtype=torch.float)
        if x.is_cuda:
            temp_input = temp_input.cuda()
        hidden = en_hidden

        for t in range(frames):
            temp_input, hidden = self.decoder(temp_input, hidden)
            temp_input = self.fc(temp_input)
            reconstruct_output.append(temp_input)
        reconstruct_output = torch.cat(reconstruct_output, dim=1)

        return reconstruct_output

class LSTM_AE1(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super(LSTM_AE1, self).__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=4,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )
        self.en_relu = nn.ReLU()

        self.decoder = nn.LSTM(
            input_size=64,
            hidden_size=output_dim,
            num_layers=4,
            dropout=0,
            batch_first=True,
            bidirectional=False,
        )
        self.de_relu = nn.ReLU()
        self.fc = nn.Linear(output_dim, output_dim)
    def forward(self, x):
        frames = x.size(1)
        output, (h_n, c_n) = self.encoder(x)
        h_n = self.en_relu(h_n[-1, :, :])
        #print(h_n.size())
        h_n = h_n.unsqueeze(1).repeat(1, frames, 1)
        output, (_, _) = self.decoder(h_n)
        output = self.de_relu(output)
        output = self.fc(output)

        return output

class Auto_encoder_small(nn.Module):
    def __init__(self, input_dim=640, output_dim=640):
        super(Auto_encoder_small, self).__init__()
        self.encoder = nn.Sequential(
            Liner_Module(input_dim=input_dim, out_dim=128),
            Liner_Module(input_dim=128, out_dim=64),
            Liner_Module(input_dim=64, out_dim=32),
            Liner_Module(input_dim=32, out_dim=16),
        )
        self.decoder = nn.Sequential(
            Liner_Module(input_dim=16, out_dim=32),
            Liner_Module(input_dim=32, out_dim=64),
            Liner_Module(input_dim=64, out_dim=128),
            nn.Linear(128, output_dim),
        )

    def forward(self, input: torch.Tensor):
        x_feature = self.encoder(input)
        x = self.decoder(x_feature)
        return x, x_feature

class Auto_encoder_CNN(nn.Module):
    def __init__(self):
        super(Auto_encoder_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 7), stride=(2, 1), padding=1),
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, kernel_size=(3, 7), stride=(2, 1), padding=1),
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 7), stride=(2, 1), padding=1),
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 7), stride=(2, 1), padding=1),
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, inputs):
        x_feature = self.encoder(inputs)
        x = self.decoder(x_feature)
        return x, x_feature

if __name__ == '__main__':
    # ae = Auto_encoder(input_dim=640, out_dim=640)
    # x = torch.rand((64, 512)).float()
    # # with SummaryWriter(log_dir='./model_logs', comment='Auto_Encoder_CNN') as w:
    # #     w.add_graph(ae, x)
    # ae.cuda(0)
    #summary(ae, input_size=(640, ))
    # f = ae(x)
    # print(f.size())

    # ae = RPAE(input_dim=256, output_dim_p=64, output_dim_r=256)
    # x = torch.rand((64, 256)).float()
    # r, p = ae(x)
    # print(r.size(), p.size())
    # a = torch.tensor([[1, 2], [3, 4]])
    # b = torch.tensor([[1,2], [2,1]])
    # print(a*b)
    # net = nn.LSTM(input_size=64, hidden_size=128, num_layers=16)
    # for name, parameters in net.named_parameters():
    #     print(name, ':', parameters.size())

    a = torch.rand((64, 5, 128), dtype=torch.float32)
    net = LSTM_AE()
    b = net(a)
    print(b.size())
    # encoder = nn.LSTM(
    #     input_size=128,
    #     hidden_size=64,
    #     num_layers=4,
    #     dropout=0.1,
    #     batch_first=True,
    #     bidirectional=False,
    # )
    # decoder = nn.LSTM(
    #     input_size=128,
    #     hidden_size=64,
    #     num_layers=4,
    #     dropout=0.1,
    #     batch_first=True,
    #     bidirectional=False,
    # )
    # out, hidden = encoder(a)
    # b = torch.rand((64, 1, 128), dtype=torch.float32)
    # out, hidden = decoder(b, hidden)


