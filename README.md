# PI-NAS: 基于 AI 的阵列天线综合最小实现

本项目实现一个可运行的第一版实验：

- 8x8 均匀平面阵，阵元间距 `0.5 lambda`
- MLP 输入目标波束需求 `[theta0, phi0, SLL]`
- MLP 输出每个阵元的幅度和相位
- PyTorch 可微阵列因子计算
- Physics-informed loss 直接约束主瓣指向和旁瓣电平
- 输出训练日志、阵元权值、方向图数据和 SVG 方向图

## 运行环境

使用你指定的 conda 环境：

```powershell
D:\anaconda\envs\langchain-env\python.exe --version
D:\anaconda\envs\langchain-env\python.exe -c "import torch; print(torch.__version__)"
```

当前代码不依赖 matplotlib，方向图使用标准库生成 SVG。

## 快速运行

训练一个目标角为 30 度、旁瓣目标为 -20 dB 的 8x8 阵列：

```powershell
D:\anaconda\envs\langchain-env\python.exe .\pinas_train.py --theta0 30 --epochs 1000 --target-sll-db -20
```

一次性运行三个扫描角实验：

```powershell
D:\anaconda\envs\langchain-env\python.exe .\pinas_train.py --sweep -30 0 30 --epochs 1000 --target-sll-db -20
```

输出文件保存在 `outputs/`，包括：

- `pattern_theta_*.svg`：方向图
- `pattern_theta_*.csv`：角度与归一化方向图数据
- `weights_theta_*.csv`：阵元幅度和相位
- `summary.csv`：实验指标汇总

## 方法说明

网络不是监督学习，不需要提前生成“方向图-权值”标签数据。训练时网络输出阵元权值，再通过可微阵列因子计算方向图，最后用物理约束损失反向传播更新网络参数。

核心损失包括：

- 主瓣指向损失：目标角方向增益应接近全局峰值
- 旁瓣约束损失：主瓣保护区外的方向图超过目标 SLL 时才惩罚，默认使用 top-k 最高旁瓣点
- 幅度正则：避免幅度全部塌缩或过度稀疏

默认 `--mainlobe-guard-deg 20` 是针对 8x8 小阵列和 ±30 度扫描的保守设置。后续扩展到更大阵列时，主瓣会变窄，可以相应减小这个角度。

这对应“MLP 网络结构 + Physics-informed Loss + 可微阵列因子”的第一阶段实现。
