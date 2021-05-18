from torch import nn

"""
    Dueling Networkを用いたQ関数を実現するためのニューラルネットワークをクラスとして記述します. 
"""
class CNNQNetwork(nn.Module):
    def __init__(self, state_shape, n_action):
        super().__init__()
        self.state_shape = state_shape
        self.n_action = n_action

        if (state_shape[1] == 10):
          kernel_size1 = 3
          stride1 = 1
          stride2 = 1
        elif (state_shape[1] == 32):
          kernel_size1 = 4
          stride1 = 2
          stride2 = 1
        elif (state_shape[1] == 64):
          kernel_size1 = 4
          stride1 = 2
          stride2 = 2
        else:
          raise Exception("Please choose board size N from 10,32,64 when you use the CNN for policy network!!")

        # Dueling Networkでも, 畳込み部分は共有する
        self.conv_layers = nn.Sequential(
            nn.Conv3d(state_shape[0], 32, kernel_size=kernel_size1, stride=stride1),  # 1xNxN -> 32 x next_layer_n x next_layer_n
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=stride2),  # 32x next_layer_n x next_layer_n -> 64 x convoluted_n x convoluted_n
            nn.ReLU(),
        )

        cnn_out_size = self.check_output_size(state_shape, self.conv_layers) # CNN共有層にかけた後の出力層の次元を解析

        # Dueling Networkのための分岐した全結合層
        # 状態価値
        self.fc_state = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # アドバンテージ
        self.fc_advantage = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )


    def forward(self, obs):
        feature = self.conv_layers(obs)
        feature = feature.view(feature.size(0), -1)  #　Flatten. 64x7x7　-> 3136

        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature)

        # 状態価値 + アドバンテージ で行動価値を計算しますが、安定化のためアドバンテージの（行動間での）平均を引きます
        action_values = state_values + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return action_values
    
    def check_output_size(self, shape, net):
        shape = torch.FloatTensor(1,shape[0],shape[1],shape[2],shape[3])
        out = net(shape).size()
        out = np.prod(np.array(out))
        return out