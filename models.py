import torch
import torch.nn as nn


class DeepKoopmanExperiment1(nn.Module):
    def __init__(self, params):
        super(DeepKoopmanExperiment1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(params['indim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['obsdim']),
        )

        self.decoder = nn.Sequential(
            nn.Linear(params['obsdim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['outdim']),
        )

        self.lr = nn.Linear(params['obsdim'], params['n_targets'], bias=True)

    def forward(self, x0):
        y = self.encoder(x0)
        x1 = self.decoder(y)
        return x1

class DeepKoopmanExperiment2(nn.Module):
    def __init__(self, params):
        super(DeepKoopmanExperiment2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(params['indim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['obsdim']),
        )

        self.decoder = nn.Sequential(
            nn.Linear(params['obsdim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['outdim']),
        )

        self.lr = nn.Linear(params['obsdim'], params['n_targets'], bias=True)

    def forward(self, x0):
        y = self.encoder(x0)
        x1 = self.decoder(y)
        return x1

class DeepKoopmanExperiment3(nn.Module):
    def __init__(self, params):
        super(DeepKoopmanExperiment3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(params['indim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['obsdim']),
        )

        self.K = nn.Linear(params['obsdim'], params['obsdim'], bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(params['obsdim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['outdim']),
        )

        self.lr = nn.Linear(params['obsdim'], params['n_targets'], bias=True)

    def forward(self, x0):
        y0 = self.encoder(x0)
        y1 = self.K(y0)
        x1 = self.decoder(y1)
        return x1





class DeepKoopman3(nn.Module):
    def __init__(self, params):
        super(DeepKoopman3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(params['indim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['obsdim']),
        )

        self.K = nn.Linear(params['obsdim'], params['obsdim'], bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(params['obsdim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['outdim']),
        )

        self.lr = nn.Linear(params['obsdim'], params['n_targets'], bias=True)

    def forward(self, x0):
        yk0 = self.encoder(x0)
        yk1 = self.K(yk0)
        x1 = self.decoder(yk1)
        return x1