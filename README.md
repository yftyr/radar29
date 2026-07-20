# PI-NAS: 基于 AI 的阵列天线综合

本项目实现“MLP 网络结构 + Physics-informed Loss + 可微阵列因子”的阵列天线综合流程，并扩展出一个面向比赛指标的版本。

## 环境

使用指定 conda 环境：

```powershell
D:\anaconda\envs\langchain-env\python.exe --version
D:\anaconda\envs\langchain-env\python.exe -c "import torch, matplotlib; print(torch.__version__); print(matplotlib.__version__)"
```

## 第一阶段：8x8 最小闭环

该脚本用于验证 AI 自动输出阵元幅相权值的基本闭环。

```powershell
D:\anaconda\envs\langchain-env\python.exe .\pinas_train.py --sweep -30 0 30 --epochs 1000 --target-sll-db -20
```

输出目录：`outputs/`

## 比赛扩展版：32x32 千阵元综合

`competition_pinas.py` 面向比赛要求加入了：

- 32x32 均匀平面阵，默认 1024 阵元，满足 `N >= 1000`
- 默认支持 `-60 到 60` 度扫描范围内的目标角
- 和波束综合，默认目标副瓣 `<= -35 dB`
- 差波束综合，默认目标零深 `<= -30 dB`，差波束副瓣 `<= -20 dB`
- 自适应置零，默认 4 个零陷：`-45,-20,20,45`
- 阵元失效仿真，通过 `--fault-rate` 设置 5%-20% 失效率
- Matplotlib 输出和/差波束对比图
- CSV 输出方向图、阵元权值和指标汇总

快速验证一个 32x32 目标：

```powershell
D:\anaconda\envs\langchain-env\python.exe .\competition_pinas.py --theta0 30 --epochs 800
```

验证 ±60 度扫描：

```powershell
D:\anaconda\envs\langchain-env\python.exe .\competition_pinas.py --sweep -60 -30 0 30 60 --epochs 800
```

加入 10% 阵元失效：

```powershell
D:\anaconda\envs\langchain-env\python.exe .\competition_pinas.py --theta0 30 --fault-rate 0.10 --epochs 240
```

输出目录：`outputs_competition/`

## 方法说明

比赛版不是纯随机优化。网络输出的是对物理基线的残差修正：

```text
目标需求
  -> MLP
  -> 和波束/差波束幅相残差
  -> 物理基线权值 + AI 残差
  -> 可微阵列因子
  -> 多目标 Loss
  -> AdamW 反向传播
```

多目标损失包括：

- 和波束主瓣指向损失
- 和波束最高旁瓣 top-k 约束
- 指定角度零陷约束
- 差波束目标角零深约束
- 差波束左右瓣形成约束
- 差波束旁瓣 top-k 约束

默认 `--nulls auto4` 会根据当前扫描角自动选择 4 个避开主瓣的零陷角，避免把零陷放进目标主瓣造成约束冲突。默认和/差波束保护区还会随扫描角按投影展宽自动放大，适配 ±60 度扫描。

这相当于把前期 8x8 的“能跑通”模型升级为比赛要求导向的“大规模、多约束、可验证”模型。
