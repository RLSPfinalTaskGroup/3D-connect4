import random
import torch
import numpy as np
from abc import ABC, abstractmethod

class Algorithm(ABC):
  @abstractmethod
  def step(self, env, obs, t):
    """ 環境(env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
        受け取り，リプレイバッファへの保存などの処理を行い，状態・エピソードのステップ数を更新する．
    """
    pass


  @abstractmethod
  def is_update(self):
    """ 現在のトータルのステップ数(steps)を受け取り，アルゴリズムを学習するか否かを返す． """
    pass


  @abstractmethod
  def update(self):
    """ 1回分の学習を行う． """
    pass


  def random_act(self, env):
    top_layer = np.array(env.board)[env.num_grid-1, :, :]
    indices = np.where(top_layer==0)
    selected_index = random.randrange(len(indices[0]))
    action = env.num_grid*indices[0][selected_index] + indices[1][selected_index]
    return action


  # epsilon-greedy. 確率epsilonでランダムに行動し, それ以外はニューラルネットワークの予測結果に基づいてgreedyに行動
  def act(self, obs, env, epsilon):
    if random.random() < epsilon:
      action = self.random_act(env)
    else:
      # 行動を選択する時には勾配を追跡する必要がない
      with torch.no_grad():
        action = torch.argmax(self.model(obs.unsqueeze(0))).item()
    return action