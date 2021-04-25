# dqn-tetris

## 開発環境
- python：3.8.5
- その他外部モジュール：conda_requirements.txtを参照

## 実行方法
- 訓練
    - experiment-nameに対応する設定ファイル（configs/"experiment-name".yaml）が読み込まれ訓練が開始されます。
    - 訓練前に設定ファイル（"experiment-name".yaml）を作成しておく必要があります。
    - 標準出力のログ（"experiment-name".log）、stepなどを記録したcsv（results/"experiment-name".csv）、最もstepが続いた際の重みファイル（"experiment-name".csv）、stepなどの遷移グラフ（"experiment-name"_step.png）などが保存されます。
    ```
    python train.py "experiment-name"
    ```
- 推論
    - experiment-nameに対応するモデルの重みファイル（models/"experiment-name".pth）が読み込まれ推論が行われます。<br>
    - Javaのテトリスで使用する重みファイル（models/tetrisnet.pt）、推論時の様子のgifファイル（results/"experiments-name".gif）が出力されます。
    ```
    python test.py "experiment-name"
    ```

## 改善案
- 現在一番うまくいった方法は総合演習から報酬を（消したラインの数+1）（2乗をしない）とし、学習率のスケジューラにMultiSteplrを用いたものです。
- エポック数は5000です。最初10000する設定でしたが、steplr.logにもある通り4000付近で最高ステップ数が4000回程度に更新された後6000ステップまで更新がなかったこと、１ステップの長さが飛躍的に伸びたため１ステップの時間がかなり長くなるようになったためです。
- よりステップ数を伸ばす方法としては以下のようなものが考えられる。
    - 画像サイズとモデルサイズを拡大する。改善する可能性が大きいがJava側での入力も合わせて整形する必要がある。なお、現在Java側ではsleep()などを使ってAI側の操作を早すぎないように調整しているため、モデルサイズを大きくしても推論時間としては全く問題ないと考えられる。
    - 次のピースの情報を入力に入れるようにする。こちらもJava側の入力を合わせて整形する必要がある。
