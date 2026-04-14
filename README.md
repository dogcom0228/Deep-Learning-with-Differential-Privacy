# Deep Learning with Differential Privacy

這個專案已重構為以 PyTorch 與 Opacus 為核心的現代化 DP-SGD 實驗框架，目標是重新實作 Deep Learning with Differential Privacy 論文中的核心訓練流程，同時符合目前的工程實務與 GPU 執行環境。

## 重構目標

- Python 3.12+
- Conda 環境名稱固定為 dp-sgd
- PyTorch + torchvision + Opacus
- 面向 RTX 30 / 40 / 50 系列顯卡
- 預設採用 CUDA 13.0 wheel，適用於 CUDA 12.9+ 驅動環境
- 結構化、模組化、可重複實驗

## 專案結構

```text
.
├── configs/
│   ├── cifar10-dp.yaml
│   ├── cifar10-sgd.yaml
│   ├── mnist-dp.yaml
│   └── mnist-sgd.yaml
├── src/dp_sgd/
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── optim.py
│   ├── privacy.py
│   ├── train.py
│   └── utils.py
├── cifar10.py
├── mnist.py
├── dp_optimizer.py
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

## 建立環境

```bash
conda env create -f environment.yml
conda activate dp-sgd
pip install -e . --no-deps
```

說明：

- environment.yml 會建立 Python 3.12 的 conda 環境，並透過 requirements.txt 安裝 PyTorch CUDA 13.0 wheel 與 Opacus。
- pip install -e . --no-deps 只會把本專案以 editable mode 掛進環境，不會覆寫已安裝的 GPU 版 PyTorch。

## 執行方式

直接跑預設入口：

```bash
python mnist.py
python cifar10.py
```

使用統一 CLI：

```bash
python -m dp_sgd train --config configs/mnist-dp.yaml
python -m dp_sgd train --config configs/cifar10-dp.yaml
```

常見覆寫：

```bash
python mnist.py --epochs 10 --override privacy.noise_multiplier=1.2
python cifar10.py --non-private --device cuda
```

## 實作重點

- 使用 Opacus PrivacyEngine 實作 DP-SGD，而不是舊版 tensorflow-privacy optimizer fork
- 模型預設採用 GroupNorm，避免 BatchNorm 在 DP 訓練中的相容性問題
- 啟用 TF32、channels_last、pin_memory、persistent_workers 等現代化 PyTorch 執行設定
- 非 DP baseline 可使用 torch.compile 與 AMP
- 每次訓練會輸出 resolved-config.yaml、history.csv、metrics.json、best.pt、last.pt

## 結果輸出

所有實驗產物會寫入 results/<experiment-name>/<timestamp>/：

- resolved-config.yaml
- history.csv
- metrics.json
- best.pt
- last.pt

## 舊版程式碼

根目錄保留的 tensorflow_privacy/ 目錄僅作為歷史參考，不再是主要訓練路徑。新的訓練入口已全面切換到 PyTorch / Opacus。
