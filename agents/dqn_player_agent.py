class PlayerDQN(Algorithm):
  def __init__(self, model, target_model, loss_func, optimizer, buffer,
               beta_begin=0.4, beta_end=1.0, beta_decay=1000, 
               epsilon_begin = 1.0, epsilon_end = 0.01, epsilon_decay = 500000,
               gamma=0.99, batch_size=64, initial_buffer_size=100, eps_for_eval=0.05, seed=0):
    # seed値の設定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    self.model = model
    self.target_model = target_model
    self.loss_func = loss_func
    self.optimizer = optimizer
    self.buffer = buffer
    
    self.gamma = gamma
    self.batch_size = batch_size
    self.initial_buffer_size = initial_buffer_size
    self.eps_for_eval = eps_for_eval
    
    # epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす
    self.epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))
    
    # beta_beginから始めてbeta_endまでbeta_decayかけて線形に増やす
    self.beta_func = lambda step: min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))


  def step(self, env, obs, t, step_total, is_eval=False):
    this_agent_done = False
    
    t += 1 # エピソードステップを＋1
    
    # ε-greedyのためのεの定義。訓練時は総ステップ数に応じて線形減少させる
    eps = self.epsilon_func(step_total)
    # 評価時は固定値
    if (is_eval):
      eps = self.eps_for_eval

    self.curr_action = self.act(obs.to(device), eps) # ε-greedyで行動を選択
    next_obs, reward, done, info = env.step(self.curr_action) # 環境中で実際に行動

    # 置けない場所におこうとした場合、その情報はバッファに格納され、もう一度自分のターン（置き直せる）
    # ただ、環境から帰ってくる負の報酬（penalty）はステップ報酬に（従ってエピソード報酬・総報酬和にも）計上される。
    # また、「プレーヤーが変わらない」という処理は環境側で行っていることに注意。
    if (info["is_couldnt_locate"]==True):      
      # 評価時には経験の蓄積は行わない
      if (not is_eval):
        self.push_to_buffer(obs, reward, next_obs, done) # 置けなかったときのことを学習させる（経験バッファに格納）
      
      t -= 1 # 次も自分のターン（置き直し）なので、エピソードステップを1戻しておく。
  
    # 置ける場所に置いた場合
    else:
      this_agent_done =True # このエージェントの行動が終了したことを意味するフラグをオンにする

    # もしこのエージェントの行動により、ゲーム自体（エピソード）が終了していた場合
    if (done):
      t = 0 # エピソードステップを初期化

    return next_obs, reward, done, info, t, this_agent_done


  def is_update(self):
    return len(self.buffer) > self.initial_buffer_size


  def update(self, step_total):
    obs, action, reward, next_obs, done, indices, weights = self.buffer.sample(self.batch_size, self.beta_func(step_total))
    obs, action, reward, next_obs, done, weights \
        = obs.to(device), action.to(device), reward.to(device), next_obs.to(device), done.to(device), weights.to(device)

    #　ニューラルネットワークによるQ関数の出力から, .gatherで実際に選択した行動に対応する価値を集めてきます.
    q_values = self.model(obs).gather(1, action.unsqueeze(1)).squeeze(1)

    # 目標値の計算なので勾配を追跡しない
    with torch.no_grad():
      # Double DQN. 
      # ① 現在のQ関数でgreedyに行動を選択し, 
      greedy_action_next = torch.argmax(self.model(next_obs), dim=1)

      # ②　対応する価値はターゲットネットワークのものを参照します.
      q_values_next = self.target_model(next_obs).gather(1, greedy_action_next.unsqueeze(1)).squeeze(1)

    # ベルマン方程式に基づき, 更新先の価値を計算します.
    # (1 - done)をかけているのは, ゲームが終わった後の価値は0とみなすためです.
    target_q_values = reward + gamma * q_values_next * (1 - done)

    # Prioritized Experience Replayのために, ロスに重み付けを行なって更新します.
    self.optimizer.zero_grad()
    loss = (weights * self.loss_func(q_values, target_q_values)).mean()
    loss.backward()
    self.optimizer.step()

    #　TD誤差に基づいて, サンプルされた経験の優先度を更新します.
    self.buffer.update_priorities(indices, (target_q_values - q_values).abs().detach().cpu().numpy())

    return loss.item()

    
  # bufferの処理はエージェントクラス側に一任。Trainerからも呼ぶ必要があったので、関数として用意した。
  # Trainer側は「エージェントがどのようなactionをとったか」について関知しないので、actionのみ（引数でなく）クラスのインスタンス変数を使っている。
  def push_to_buffer(self, obs_before_act, reward, obs_afeter_act, done):
    return self.buffer.push([obs_before_act, self.curr_action, reward, obs_afeter_act, done])