# ソーシャルVR空間に適用可能な好奇心ベースの自律機械知能

![banner](/docs/images/banner.png)

## バーチャル学会 2024 発表概要

<https://vconf.org/2024/poster/d2/#3>

## メンバー

- GesonAnko
- myxy
- zassou
- ぶんちん
- 田中スイセン
- Klutz

## 論文を読んできてくれた方へ

論文を読んでいただき誠にありがとうございます。以下に論文中で紹介している機能の実装がありますのでご確認ください。

### 経験データのバッファ実装

<img src="docs/images/data_buffer_move.svg" width="300" alt=data_buffer_move>

- [BaseDataBuffer](/ami/data/buffers/base_data_buffer.py)
- [DataCollector,DataUser](/ami/data/interfaces.py)

### 学習スレッドと推論スレッド間のモデルパラメータ同期

<img src="docs/images/model_sync.svg" width="300" alt=model_sync>

- [BaseTrainer.\_sync_a_model](/ami/trainers/base_trainer.py#L192)

### Unity上のワールド

以下のリポジトリに別途実装しました。

<https://github.com/MLShukai/AMIUnityEnvironment>

### Join the Discord Server !

自律機械知能P-AMI\<Q>の開発について詳しく知りたい方はぜひDiscordサーバへ！

<https://discord.gg/54EhvtZgHH>

## 開発環境セットアップ

### プロジェクト単体

事前にPoetryを導入しておく。

```sh
poetry install
poetry run python script/launch.py
```

Poetryの基本的な使い方は[Poetryをサクッと使い始めてみる](https://qiita.com/ksato9700/items/b893cf1db83605898d8a)を参照。

### VRChat連携

<!-- ここにVRChat連携の図を貼る -->

#### ハードウェア要件

| ハードウェア | 要件                        | 備考                                                   |
| ------------ | --------------------------- | ------------------------------------------------------ |
| CPU          | Intel / AMD の64bit CPU     |                                                        |
| GPU          | NVIDIA GPU                  | AMD Radeonでも技術的には対応可能と思われるが、未検証。 |
| ディスプレイ | FHD以上の画質のディスプレイ | VRChatの起動のため。                                   |

#### ソフトウェア要件

- Linux Desktop OS、特に Ubuntu Desktop 22.04 LTS以降。（他のLinux OSでも依存アプリケーション（VRChatなど）が正常にインストールできれば問題ない。）

  **NOTE**: **VRChatやOBSを起動するためにはDesktop環境が必須**

  **NOTE**: 比較的新しいGPUに対応するため、直近のUbuntuリリースを使うことが望ましい。Ubuntu 22.04ではGeforce RTX 3060を認識せず、インストール時に画面表示ができない不具合があった。23.10にバージョンを変更することで解決した。

- 最新の安定版 NVIDIA Driver（2024/03/23 時点では version **550**）

  Ubuntuの場合は `sudo ubuntu-drivers install`でハードウェアに最も適したドライバーが自動でインストールされる。

  参考: [NVIDIA drivers installation | Ubuntu Server](https://ubuntu.com/server/docs/nvidia-drivers-installation)

### VRChatとの連携

1. Steamをインストール。

   <https://store.steampowered.com/about/> からDEBファイルをダウンロードし、 `sudo dpkg -i steam_latest.deb; sudo apt install --fix` を実行。

   **NOTE**: NVIDIA Driverのバージョン**525以下**ではSteamが正常に動作しないため注意。（提供されている最新の安定版Driverを使うこと。）

   **NOTE**: Snap版のSteamではVRChatをインストールできないため、使わないこと。Steamの公式ページからdebファイルをダウンロードしインストールする。

2. Steamの設定で「互換性」「他のすべてのタイトルでSteam Playを有効化」をオンにする。

   ![image](https://github.com/MLShukai/ami/assets/574575/c5d3d36f-fc28-44e0-87d6-e6c5113bb1b0)

3. VRChatをインストール。

[vrchat-ioのドキュメンテーション](https://github.com/Geson-anko/vrchat-io?tab=readme-ov-file#vrchat)を参考に、VRChatやOBSをインストールする。

### Docker

VRChatやOBSなどのホストOSに依存したものを除いた、Pythonなどの開発環境はDockerイメージにまとめてある。

Linux OSに事前に次のツールをインストールしておく。

- Docker

  Dockerの公式ドキュメンテーションを参考にインストールを行う。

  [Install Docker Engine](https://docs.docker.com/engine/install/)

  `sudo` なしでDockerを操作するためには以下の手順を実行する。

  [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/)

- NVIDIA Container Toolkit

  NVIDIA Container Toolkitをインストールする。

  [Installing with Apt](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)

  コンテナラインタイムをDocker用に設定する。

  [Configuring Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)

- make

  Ubuntu Desktopではプリインストールされていないため、 `sudo apt install build-essentials` でインストールする。

次のコマンドでイメージをビルドし、起動する。

```sh
# project root. (ami/)
make docker-build
make docker-run
```

後は、VSCodeなどのエディタからDockerコンテナにアタッチし、 `/workspace` ディレクトリで作業を行う。このディレクトリはホストOSの永続ボリュームであるため、Dockerのコンテナインスタンスを削除しても作業内容はホストOSに保存される。

## Wiki

追加のドキュメントは[Wiki](https://github.com/MLShukai/ami/wiki)に整備予定。
