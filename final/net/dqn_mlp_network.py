import torch
from torch import nn

"""
    Dueling Networkを用いたQ関数を実現するためのニューラルネットワークをクラスとして記述します.
"""
class MLPQNetwork(nn.Module):
    def __init__(self, state_shape, n_action):
        super().__init__()
        self.state_shape = state_shape
        self.n_action = n_action
        # 共通ネットワーク部分
        self.share_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_shape[0]*state_shape[1]*state_shape[2], 512),
            nn.LeakyReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        )

        # Dueling Networkのための分岐した全結合層
        # 状態価値
        self.fc_state = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        # アドバンテージ
        self.fc_advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, n_action)
        )


    def forward(self, obs):
        feature = self.share_layers(obs)

        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature)

        # 状態価値 + アドバンテージ で行動価値を計算しますが、安定化のためアドバンテージの（行動間での）平均を引きます
        action_values = state_values + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return action_values