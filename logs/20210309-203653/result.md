# Results of experiments

## Results
- 50000episodes, 720000stepsほど学習させた
- 勝率は50％から徐々に増大
- couldn't locateの回数はstepを重ねるごとに増えた
    - 同時にEpisodeRewardも減少
- lossは安定していない（あまり変化がない）
```
Result (100times)
-------------------
win_rate: 73.0
draw_rate: 0.0
lose_rate 27.0
sum: 100.0
```

## Discussion 
- couldn't locateの回数が徐々に増えていく原因
  - 適切な学習が行われていない状況でgreedyにactionを選択する確率(1-ε)が増えているから
- 総合的に見て学習が少しいい方向へ
  - パラメータを増やせば増やすほど良いという文献(NLPに関する)を見て、試したところ良い結果となった

## task 
- さらにパラメータを増やしてみる




