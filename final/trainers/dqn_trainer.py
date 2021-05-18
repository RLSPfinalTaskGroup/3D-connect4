# エージェントと環境をメンバーとして持つ中間管理職
from .abstract_trainer import Trainer
import torch
import numpy as np
import matplotlib.pyplot as plt

class DQNTrainer(Trainer):

  def __init__(self, env, test_env, player_agent, enemy_agent, first_player, writer, weight_folder,
              #  seed=0,
               target_update_interval=200, model_save_interval=4000, num_episodes=10**5, eval_interval=10**4, num_eval_episodes=3):
    self.env = env
    self.test_env = test_env # 環境
    self.player_agent = player_agent
    self.enemy_agent = enemy_agent
    self.first_player = first_player # 先行はどちらか
    self.writer = writer
    self.weight_folder = weight_folder

    # 環境の乱数シードを設定する．
    # self.env.seed(seed)
    # self.test_env.seed(2**31-seed)

    # 平均収益を保存するための辞書．
    self.returns = {'episode': [], 'return': []}

    self.target_update_interval = target_update_interval # 学習安定化のために用いるターゲットネットワークの同期間隔
    self.model_save_interval = model_save_interval # networkの重みを保存する間隔

    # 何エピソード学習を行うか．
    self.num_episodes = num_episodes
    # 何エピソードごとに評価を行うか．
    self.eval_interval = eval_interval
    # 一度の評価で、何エピソード分評価するか．
    self.num_eval_episodes = num_eval_episodes


  def train(self):
    info={"turn": self.first_player, "winner": 0}

    step_total = 0 # 実験を通して今何ステップ目か
    total_reward = 0 # 実験全体を通した報酬和（総報酬和）
    enemy_update = 0 # 敵エージェントの方策更新回数

    num_win=0 # 勝利数
    num_lose=0 # 敗北数
    num_draw=0 # 引き分け数

    # num_episode回ゲームを行う。
    for episode in range(self.num_episodes):
      obs = self.env.reset()
      done = False

      step_start = step_total # このゲームが何ステップ目から開始したかを初期化
      step = 0 # 現在のエピソードで何ステップ目か

      episode_reward=0 # ゲームを通した報酬の合計（エピソード報酬）

      while not done:
        sum_reward = 0 # 1ステップ分の報酬（ステップ報酬）
        player_reward = 0 # 自分のターンの間に得た報酬の和（couldnt_locateが連続した場合は、自分のターンの間に負の報酬が蓄積されるため）
        enemy_reward = 0 # 相手のターンの間に得た報酬の和（couldnt_locateが連続した場合は、相手のターンの間に負の報酬が蓄積されるため）
        is_end_player_step = False # 1ステップのうち、プレーヤーの行動が終了したかどうかの判定フラグ
        is_end_enemy_step = False # 1ステップのうち、相手の行動が終了したかどうかの判定フラグ
        num_couldnt_locate = 0 # 一定回数以上おけないところに置こうとしたら、強制的に空いているところに玉を置くようにするためのカウンタ

        # ゲーム自体が終了(done)するか、両方のプレーヤーの手番が終了したら、1ステップ終了
        while (not done) and (not is_end_player_step or not is_end_enemy_step):
          # 自分のターンの場合
          if (info["turn"] == 1):
            next_obs, reward, done, info, step, num_couldnt_locate, is_end_player_step = self.player_agent.step(self.env, obs, step, num_couldnt_locate, step_total)
            obs_before_player_act = obs
            obs_after_player_act = next_obs
            player_done = done
            player_reward += reward

          # 相手ターンの場合
          elif (info["turn"] == -1):
            next_obs, reward, done, info, step, is_end_enemy_step = self.enemy_agent.step(self.env, obs, step)
            enemy_reward += reward

          # 次の観測をセット
          obs = next_obs

        # ステップ報酬に、自プレーヤーの得た報酬と、敵プレーヤーの得た報酬（自プレーヤーからみた場合は罰）を計上。
        # 今は、sum_reward は player_agent 視点の「報酬」を意味しているので、 enemy_reward は減算処理をする。
        sum_reward = player_reward - enemy_reward

        episode_reward += sum_reward # エピソード報酬にステップ報酬を計上
        step_total += 1 # 総ステップ数を増やす

        # リプレイバッファに経験を蓄積(置けないところに置いた時との違いは、バッファに記録する経験の報酬が自プレーヤーの報酬のみを含むか"sum_reward"かという部分だけ。)
        # こちらの経験は、相手プレーヤーの行動によって勝利が生まれた場合に、その分の罰を含んだ経験となる。
        # ちなみにcouldnt_locate=True(置けないところに置いた)の時は、ここまでくることはない。（whileループを出るには　is_end_player_step=True の必要があるが、couldnt_locate=True の場合はずっと　is_end_player_step=False　のままのため）
        self.player_agent.push_to_buffer(obs_before_player_act, sum_reward, obs_after_player_act, player_done)

        # アルゴリズムが準備できていれば，1回学習を行う．(bufferに十分なデータが蓄えられていれば、DQNでは毎ステップ学習がかかる)
        if self.player_agent.is_update():
            loss = self.player_agent.update(step_total=step_total)
            # tensorboardに書き込み
            self.writer.add_scalar('Loss', loss, step_total)

      total_reward += episode_reward # 総報酬和にエピソード報酬を計上
      num_steps_in_episode = step_total - step_start # 今回のゲームが実際に何ステップかかったかを記録

      # 勝利数・敗北数・引き分け数の記入
      if (info["winner"] == 1):
        num_win += 1
      elif (info["winner"] == -1):
        num_lose += 1
      else:
        num_draw += 1

      # 一定エピソードごとにコンソールに出力
      if ((episode+1) % 500 == 0):
        print('Episode: {},  TotalStep: {}, EpisodeStep: {},  EpisodeReward: {}'.format(episode + 1, step_total, num_steps_in_episode, episode_reward))

      # 一定間隔で方策評価
      if ((episode+1) % self.eval_interval == 0):
        self.evaluate(episode=episode)

      # 訓練時の結果もtensorboardに記録しておく
      # writer.add_scalar('Total-Reward', total_reward, episode+1)
      self.writer.add_scalar('EpisodeReward', episode_reward, episode+1)
      self.writer.add_scalar('EpisodeStep', num_steps_in_episode, episode+1)
      self.writer.add_scalar('WinningRate', num_win/(episode+1) * 100, episode+1)
      self.writer.add_scalar('LosingRate', num_lose/(episode+1) * 100, episode+1)
      self.writer.add_scalar('DrawingRate', num_draw/(episode+1) * 100, episode+1)
      self.writer.add_scalar('CouldntLocateRate', num_couldnt_locate/num_steps_in_episode * 100, episode+1)

      # enemyネットワークを定期的に強くする
      if self.enemy_agent.is_update(episode):
        self.enemy_agent.update(state_dict=self.player_agent.target_model.state_dict())
        enemy_update += 1

      # ターゲットネットワークを定期的に同期させる
      if (episode + 1) % self.target_update_interval == 0:
        self.player_agent.target_model.load_state_dict(self.player_agent.model.state_dict())

      # networkの重みを定期的に保存
      if ((episode + 1) % self.model_save_interval == 0):
        torch.save(self.player_agent.model.state_dict(), self.weight_folder + "episode_{}.pth".format(episode+1))

    torch.save(self.player_agent.model.state_dict(), self.weight_folder+"weights_final.pth")
    self.writer.close()


  def evaluate(self, episode=None, is_final=False):
    """ 複数エピソード環境を動かし，平均収益を記録する． """
    eps = 0.1
    num_eval_episodes = self.num_eval_episodes
    # 全学習終了後の、ランダムな行動をとるエージェントに対しての性能比較のための分岐。
    if (is_final):
      eps = 0.99
      num_eval_episodes = 100

    info={"turn": self.first_player, "winner": 0}

    step_total = 0 # 評価実験を通して今何ステップ目か
    returns = []

    num_win = 0 # 勝利数
    num_lose = 0 # 敗北数
    num_draw = 0 # 引き分け数
    num_couldnt_locate = 0 # 置けない場所におこうとした回数

    # num_eval_episodes回ゲームを行う。
    for _ in range(num_eval_episodes):
      obs = self.test_env.reset()
      done = False

      step = 0 # 現在のエピソードで何ステップ目か

      episode_reward=0 # ゲームを通した報酬の合計（エピソード報酬）

      while not done:
        sum_reward = 0 # 1ステップ分の報酬（ステップ報酬）
        player_reward = 0 # 自分のターンの間に得た報酬の和（couldnt_locateが連続した場合は、自分のターンの間に負の報酬が蓄積されるため）
        enemy_reward = 0 # 相手のターンの間に得た報酬の和（couldnt_locateが連続した場合は、相手のターンの間に負の報酬が蓄積されるため）
        is_end_player_step = False # 1ステップのうち、プレーヤーの行動が終了したかどうかの判定フラグ
        is_end_enemy_step = False # 1ステップのうち、相手の行動が終了したかどうかの判定フラグ

        # ゲーム自体が終了(done)するか、両方のプレーヤーの手番が終了したら、1ステップ終了
        while (not done) and (not is_end_player_step or not is_end_enemy_step):
          # 自分のターンの場合
          if (info["turn"] == 1):
            next_obs, reward, done, info, step, num_couldnt_locate, is_end_player_step = self.player_agent.step(self.test_env, obs, step, num_couldnt_locate, step_total, is_eval=True)
            player_reward += reward

          # 相手ターンの場合
          elif (info["turn"] == -1):
            next_obs, reward, done, info, step, is_end_enemy_step = self.enemy_agent.step(self.test_env, obs, step, eps)
            enemy_reward += reward

          # 次の観測をセット
          obs = next_obs

        # ステップ報酬に、自プレーヤーの得た報酬と、敵プレーヤーの得た報酬（自プレーヤーからみた場合は罰）を計上。
        # 今は、sum_reward は player_agent 視点の「報酬」を意味しているので、 enemy_reward は減算処理をする。
        sum_reward = player_reward - enemy_reward

        episode_reward += sum_reward # エピソード報酬にステップ報酬を計上
        step_total += 1 # 総ステップ数を増やす

      # 勝利数・敗北数・引き分け数の記入
      if (info["winner"] == 1):
        num_win += 1
      elif (info["winner"] == -1):
        num_lose += 1
      else:
        num_draw += 1

      returns.append(episode_reward)

    winning_rate = num_win/num_eval_episodes * 100
    losing_rate = num_lose/num_eval_episodes * 100
    drawing_rate = num_draw/num_eval_episodes * 100
    couldnt_locate_rate = num_couldnt_locate/(step_total) * 100
    mean_return = np.mean(returns)

    # 対ランダム方策エージェントとの比較実験の時は、ログに追加しない。
    if (not is_final):
      self.returns['episode'].append(episode)
      self.returns['return'].append(mean_return)

      self.writer.add_scalar('Val-WinningRate', winning_rate, episode+1)
      self.writer.add_scalar('Val-LosingRate', losing_rate, episode+1)
      self.writer.add_scalar('Val-DrawingRate', drawing_rate, episode+1)
      self.writer.add_scalar('Val-CouldntLocateRate', couldnt_locate_rate, episode+1)
      self.writer.add_scalar('Val-MeanReturn', mean_return, episode+1)

    print("Win: {}%, Lose: {}%, Draw: {}%, Couldnt: {}%, Return: {}".format(winning_rate, losing_rate, drawing_rate, couldnt_locate_rate, mean_return))


  def visualize(self):
    """ 1エピソード環境を動かし，mp4を再生する． """
    env = wrap_monitor(gym.make(self.env.unwrapped.spec.id))
    state = env.reset()
    done = False

    while (not done):
      action = self.algo.exploit(state)
      state, _, done, _ = env.step(action)

    del env
    return play_mp4()


  def plot(self):
    """ 平均収益のグラフを描画する． """
    fig = plt.figure(figsize=(8, 6))
    plt.plot(self.returns['episode'], self.returns['return'])
    plt.xlabel('Episodes', fontsize=24)
    plt.ylabel('Return', fontsize=24)
    plt.tick_params(labelsize=18)
    plt.title("{} in A Row".format(self.env.utils.num_win_seq), fontsize=24)
    plt.tight_layout()