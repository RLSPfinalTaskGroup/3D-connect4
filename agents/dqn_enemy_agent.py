class EnemyDQN(Algorithm):
  def __init__(self, model, enemy_update_interval=1000):
    self.model = model
    self.enemy_update_interval = enemy_update_interval # 敵のネットワークを更新する間隔(episodeに依存)


  # 今回は、あくまでも報酬は自プレーヤーの報酬のみを考えているので、相手の得る報酬は罰になることに注意。
  def step(self, env, obs, t, eps=0.1):
    return_reward = 0
    this_agent_done = False

    action = self.act(obs.to(device), eps) # epsでランダムな行動を選択し、1-epsでgreedyな行動選択をするエージェント
    next_obs, reward, done, info = env.step(action) # 環境中で実際に行動
    # 敵側の方策ネットは今回学習しないので、もし置けないところに置いた場合は単にもう一度置き直してもらう。
    # 敵プレーヤーに課せられるcoudlnt_locate_penaltyは自ブレーヤーには無関係なので無視。（reward は return_reward に計上しない）
    if (info["is_couldnt_locate"]==True):
      pass

    # 置ける場所に置いた場合。
    # 相手プレーヤーに与えられる could_locate_reward は自ブレーヤーには無関係なので無視。（reward は return_reward に計上しない）
    else:
      this_agent_done =True # このエージェントの行動が終了したことを意味するフラグをオンにする

    # 相手プレーヤーの勝利は、自プレーヤーの敗北を意味するので、この時だけ reward を return_reward に計上
    if (done):
      t = 0 # 相手の勝利でエピソードが終了しても、エピソードステップは初期化する
      return_reward = reward # 相手が勝利して得た報酬を return_reward に計上

    return next_obs, return_reward, done, info, t, this_agent_done


  def is_update(self, episode):
    return (episode + 1) % self.enemy_update_interval == 0


  def update(self, state_dict):
    self.model.load_state_dict(state_dict)