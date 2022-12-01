from torch import nn

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16,64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64,128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(128,256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256,256),
            nn.LeakyReLU(inplace=True)
        )
        self.prob_decoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),

            nn.Linear(64,3),
            nn.Softmax(dim=1),
        )
        self.value_decoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(128,24),
            nn.ReLU(inplace=True),

            nn.Linear(24,1),
            nn.Softsign(),
        )
    def forward(self,x):
        shared = self.encoder(x)
        return self.prob_decoder(shared), 2*self.value_decoder(shared)
