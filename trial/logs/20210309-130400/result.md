# Results of experiments

## Results
- 48000episodes, 720000stepsほど学習させた
- 勝敗の確率は終始50％から大きくずれることはなかった
- couldn't locateの回数はstepを重ねるごとに増えた
- lossはstepを重ねるごとに増えた

## Discussion 
- couldn't locateの回数が徐々に増えていく原因
  - 適切な学習が行われていない状況でgreedyにactionを選択する確率(1-ε)が増えているから
- 総合的に見て学習がうまく進んでいない
  - 適切な報酬と環境状態を与えられていない(置けなかった場合に`[環境,action,couldnt_locate_penalty,最初と同じ環境,done]`を与えていた一方、置けた場合は`[環境,action,reward,自分と相手が置いた後の環境,done]`としていたため、適切に処理が進まなかった可能性が高い)

## task 
- 環境と報酬の与え方
