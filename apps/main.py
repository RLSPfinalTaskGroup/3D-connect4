# ライブラリのインポート
import os
import pathlib
import random
import sys
import time

from datetime import datetime, timedelta, timezone

import gym
import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import torch

from gym import error, spaces, utils
from gym.utils import seeding
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from envs.three_d_connect_n import AnyNumberInARow3dEnv
from agents.dqn_enemy_agent import EnemyDQN
from agents.dqn_player_agent import PlayerDQN
from agents.dqn_replay_buffer import PrioritizedReplayBuffer
from net.dqn_cnn_network import CNNQNetwork
from net.dqn_mlp_network import MLPQNetwork
from trainers.dqn_trainer import DQNTrainer

# 現在の日本標準時を取得
JST = timezone(timedelta(hours=+9), 'JST')
now = datetime.now(JST).strftime('%Y%m%d-%H%M%S')

# 保存フォルダの準備
project_root = os.getcwd()
print(project_root)
save_folder = project_root + "logs/" + now
weight_folder = save_folder + "/weights/"
tensorboard_folder = save_folder + "/tensorboard/"

# 各自のDrive内に「/__MatsuoSeminerResearch/logs/【日付-時刻】/weights」という名前の保存フォルダを作成
os.makedirs(weight_folder, exist_ok=True)
# 各自のDrive内に「/__MatsuoSeminerResearch/logs/【日付-時刻】/tensorboard」という名前の保存フォルダを作成
os.makedirs(tensorboard_folder, exist_ok=True)

# cudaが使用可能かどうかを確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Model is training on", device)


""" gym環境の宣言 """
# 環境のハイパーパラメータの指定
num_grid = 10
num_win_seq = 4
win_reward = 10
draw_penalty = 5
could_locate_reward = 0.0
couldnt_locate_penalty = 0.2
first_player = 1

# 訓練用の環境を作成(Trainerクラスに渡される)
env = AnyNumberInARow3dEnv(
  num_grid=num_grid,
  num_win_seq=num_win_seq,
  win_reward=win_reward,
  draw_penalty=draw_penalty,
  could_locate_reward=could_locate_reward,
  couldnt_locate_penalty=couldnt_locate_penalty,
  first_player=first_player
)
# env = Conv3dObsWrapper(env) # 方策ネットにconv3dを使う場合

# 評価用の環境を作成(Trainerクラスに渡される)
test_env = AnyNumberInARow3dEnv(
  num_grid=num_grid,
  num_win_seq=num_win_seq,
  win_reward=win_reward,
  draw_penalty=draw_penalty,
  could_locate_reward=could_locate_reward,
  couldnt_locate_penalty=couldnt_locate_penalty,
  first_player=first_player
)
# test_env = Conv3dObsWrapper(test_env) # 方策ネットにconv3dを使う場合


""" リプレイバッファの宣言 """
buffer_size = 200000  #　リプレイバッファに入る経験の最大数

# リプレイバッファの指定(エージェントクラスに渡される)
replay_buffer = PrioritizedReplayBuffer(buffer_size)


""" ネットワークの宣言 """
# 方策ネットワークの指定(エージェントクラスに渡される)
main_net = MLPQNetwork(state_shape=env.observation_space.shape, n_action=env.action_space.n).to(device) # MLP方策ネット(自分メイン)
target_net = MLPQNetwork(state_shape=env.observation_space.shape, n_action=env.action_space.n).to(device) # MLP方策ネット(自分ターゲット)
enemy_net = MLPQNetwork(state_shape=env.observation_space.shape, n_action=env.action_space.n).to(device) # MLP方策ネット(相手メイン)
# main_net = CNNQNetwork(state_shape=env.observation_space.shape, n_action=env.action_space.n).to(device) # CNN方策ネット(自分メイン)
# target_net = CNNQNetwork(state_shape=env.observation_space.shape, n_action=env.action_space.n).to(device) # CNN方策ネット(自分ターゲット)
# enemy_net = CNNQNetwork(state_shape=env.observation_space.shape, n_action=env.action_space.n).to(device) # CNN方策ネット(相手メイン)


""" 強化学習エージェントの宣言 """
# エージェントクラスの必須パラメータ
loss_func = nn.SmoothL1Loss(reduction='none')  # ロスはSmoothL1loss（別名Huber loss）
optimizer = optim.Adam(main_net.parameters(), lr=1e-4)  # オプティマイザはAdam

# エージェントクラスの任意パラメータ
# リプレイバッファ用のハイパーパラメータ
beta_begin = 0.2
beta_end = 0.95
beta_decay = 10**4 - 2000

# ε-greedy用のハイパーパラメータ
epsilon_begin = 1.0
epsilon_end = 0.05
epsilon_decay = 10**4 - 2000

gamma = 0.99 # 時間割引率
batch_size = 4 # バッチサイズ
initial_buffer_size = 10**2 # 方策更新を始めるのに最低限必要な経験の数
eps_for_eval = 0.05 # 評価時に用いるεの値
seed = 1

enemy_update_interval = 10**3 # 敵のネットワークを更新するエピソード間隔

# 自プレーヤーと相手プレーヤーの作成(Trainerクラスに渡される)
player_agent = PlayerDQN(
    model=main_net,
    target_model=target_net,
    loss_func=loss_func,
    optimizer=optimizer,
    buffer=replay_buffer,
    device=device,
    beta_begin=beta_begin,
    beta_end=beta_end,
    beta_decay=beta_decay,
    epsilon_begin = epsilon_begin,
    epsilon_end = epsilon_end,
    epsilon_decay = epsilon_decay,
    gamma=gamma,
    batch_size=batch_size,
    initial_buffer_size=initial_buffer_size,
    eps_for_eval=eps_for_eval,
    seed=seed
    )
enemy_agent = EnemyDQN(
    model=enemy_net,
    device=device,
    enemy_update_interval=enemy_update_interval
)


""" Trainerの宣言 """
# Trainerクラスの必須パラメータ
first_player = 1
writer = SummaryWriter(log_dir=tensorboard_folder)
# writer.add_graph(net, obs.float().to(device).unsqueeze(0))

# Trainerクラスの任意パラメータ
target_update_interval=2 * 10**2 # 学習安定化のために用いるターゲットネットワークの同期間隔
model_save_interval=4 * 10**2 # networkの重みを保存する間隔
num_episodes=10**4
eval_interval=10**3
num_eval_episodes=20

trainer = DQNTrainer(
    env=env,
    test_env=test_env,
    player_agent=player_agent,
    enemy_agent=enemy_agent,
    first_player=first_player,
    writer=writer,
    # seed=seed,
    target_update_interval=target_update_interval,
    model_save_interval=model_save_interval,
    num_episodes=num_episodes,
    eval_interval=eval_interval,
    num_eval_episodes=num_eval_episodes
)


""" 学習 """
trainer.train()
trainer.plot()


""" ランダムな行動をとる敵に対する勝率の計算 """
trainer.evaluate(is_final=True)
