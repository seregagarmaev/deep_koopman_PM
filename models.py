import torch
import torch.nn as nn


class DeepKoopman(nn.Module):
    def __init__(self, params):
        super(DeepKoopman, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(params['indim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['obsdim'])
        )

        self.K = nn.Linear(params['obsdim'], params['obsdim'], bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(params['obsdim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['outdim'])
        )

        self.lr = nn.Linear(params['obsdim'], params['n_targets'], bias=True)

    def forward(self, x0):
        yk0 = self.encoder(x0)
        yk1 = self.K(yk0)
        x1 = self.decoder(yk1)
        return x1

class DeepKoopmanControl(nn.Module):
    def __init__(self, params):
        super(DeepKoopmanControl, self).__init__()
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

        self.B = nn.Sequential(
            nn.Linear(params['controldim'], params['hidden_dim']),
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

    def forward(self, x0, u0):
        yk0 = self.encoder(x0)
        Bu0 = self.B(u0)
        yk1 = self.K(yk0 + Bu0)
        x1 = self.decoder(yk1)
        return x1