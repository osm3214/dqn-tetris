# dqn-tetris

## 開発環境
- python：3.8.5
- その他外部モジュール：conda_requirements.txtを参照

## 実行方法
- 訓練
    - experiment-nameに対応する設定ファイル（configs/<experiment-name>.yaml）が読み込まれ訓練が開始されます。<br>
    - 訓練前に設定ファイルを作成しておく必要があります。<br>
    - 標準出力のログ（<experiment-name>.log）、stepなどを記録したcsv（results/<experiment-name>.csv）、最もstepが続いた際の重みファイル（<experiment-name>.csv）、stepなどの遷移グラフ（<experiment-name>_step.png）などが保存されます。
    ```
    python train.py <experiment-name>
    ```
- 推論
    - experiment-nameに対応するモデルの重みファイル（models/<experiment-name>.pth）が読み込まれ推論が行われます。<br>
    - Javaのテトリスで使用する重みファイル（models/tetrisnet.pt）、推論時の様子のgifファイル（results/<experiments-name.gif>）が出力されます。
    ```
    python test.py <experiment-name>
    ```

