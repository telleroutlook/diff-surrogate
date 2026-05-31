# Phase 8 优化执行计划 — 文献驱动的下一步增量 (2026-05-31)

> 承接 `PHASE7_OPTIMIZATION_PLAN.md`。Phase 7 已落地的工作（DiffNano R-DIT 薄层后端 + `matrix_sqrt`
> 增益层失效护栏 + CrossAttn 代理两级流程 + 真 splitter EM；OpenLithoHub 厚掩模 amplitude/phase 微扰 +
> 可微形态学 MRC 投影 + warm-start/候选打分 + 拼接残差量化；DiffCFD 多项式/样条 EOS + 强耦合 FSI 隐式
> 微分 + 吹吸边界控制环境 + 容器化交叉对拍；diff-surrogate Sobolev 导数信息训练 + 点云几何编码器 +
> 不确定度触发主动学习 + 跨仓兼容矩阵）**不再重复**，本文件只规划「未来提升」。
>
> 每条任务给出：动机文献 → 具体改动 → 验收标准（DoD）。
> 进度标记：`[ ]` 待办 · `🔬` 需先做文献复算 · `⚠️` 有许可/合规约束 · `🖥️` 需 GPU/规模化资源

---

## 0. 跨仓总主线（Phase 8 的三条战略线）

Phase 7 已把「梯度可信（gradient fidelity）」提升为一等公民。Phase 8 在此基础上往前推三步：

1. **从「梯度可信」走向「不确定度可信（calibrated UQ）」** — 下游优化/反演不仅要梯度准，还要
   *知道自己什么时候不准*。2024–2026 的功能空间共形预测（functional conformal prediction）给出了
   **分布无关、有限样本的覆盖率保证**（Ma et al. CalTech/NVIDIA；IOPscience ML:S&T 2026；arXiv:2509.04623
   函数空间 split-CP）。diff-surrogate 现有 Ensemble 方差 + 不确定度触发采样，但**没有覆盖率保证**。
   Phase 8 把「校准覆盖率（calibrated coverage）」做成统一验收门，与 Phase 7 的「梯度保真度门」并列。

2. **从「toy/CPU 自测」走向「GPU 规模化 + 与开源参照交叉复算」** — 四仓 README 仍诚实标注
   *CPU-only、无 GPU 基准、toy scale、无第三方验证*。2026 年开源生态已成熟：FDTDX（JOSS 11:8912, 2026,
   多 GPU 3D AD-FDTD）、Diff-FlowFSI（CMAME 2025, JAX GPU 可微 CFD/FSI）、meent（多后端 RCWA）、
   LithoBench/ICCAD13（ILT 公开基准）。Phase 8 收尾目标：每仓打通**真实 GPU 前向路径**并与至少一个
   **开源参照**做交叉对拍（不再只对解析解）。

3. **从「几何感知」走向「结构保持 + 可迁移基础模型雏形」** — Geo-NeW（arXiv:2602.02788, 2026-02）证明
   把守恒律/有限元外微积分作为归纳偏置，能显著改善**强 OOD 几何**的泛化；GAOT/Poseidon/DPOT 线指向
   PDE 基础模型（pretraining + 迁移）。diff-surrogate 的几何算子已能编码点云，但解算侧缺**结构保持**与
   **跨任务预训练/迁移**证据。Phase 8 给共享底座补上这两块。

---

## 1. DiffNano — 电磁求解器与反设计

**Phase 7 基线**：RCWA 四后端家族（`eig`/`eig_expm`/`matrix_sqrt`/`r_dit`，含增益层护栏）；FDTD3D 已具
**梯度检查点**（`use_checkpoint` / `checkpoint_segments`）与 device-aware 接口（默认 CPU）；
`CrossAttnRCWAProxy` + `TwoStageOptimizer` 代理预筛→RCWA 精算；splitter 已是真 S 参数 EM。

### N8.1 — FDTD 的时间反演（time-reversal）伴随：把 98% 的 AD 显存吃掉 🖥️🔬
- **动机**：*Time Reversal Differentiation of FDTD for Photonic Inverse Design*（ACS Photonics 2024，
  Hassan 组延续 2025–2026）通过在有损边界记录场、再在时间反演 FDTD 中回放，得到与伴随法同量级算力、
  但相比等价 AD **降低 98% 显存**的精确梯度（3D，900 维梯度向量实测）。DiffNano 现有的 `torch.utils.checkpoint`
  是「时间换显存」的通用手段，但仍需重算前向；时间反演伴随是 EM 专属、可在 3D 大网格上质变的路径。
- **改动**：① 在 `fdtd3d.py` 新增 `backward="time_reversal"`，记录 CPML 边界处的时变场并实现反演传播子；
  ② 与现有 `use_checkpoint` 做显存/精度对照；③ 提供 color-sorter（频域 FoM）与谐振阵列（时域 FoM）两个
  对照算例（论文同款）。
- **⚠️ 合规**：从论文 clean-room 重写，module docstring 列 references-consulted；CI 跑 `scancode`。
- **DoD**：① `backward="time_reversal"` 可用，与 AD 梯度余弦 > 0.999；② 3D 算例显存相比纯 AD 下降 ≥ 90%
  （实测表）；③ 与 `use_checkpoint` 的速度/显存权衡曲线；④ ≥ 5 新测试通过。

### N8.2 — 大面积 metasurface 的两级/局域-周期近似（LPA）+ 近场耦合修正 🖥️
- **动机**：DiffNano metalens 仍是 toy 网格（20×20…64×64）。OE 34(2):1602, 2026（*Two-level optimizer for
  large-scale metasurfaces with strong near-field coupling*）与 ACS Photonics 耦合模式理论 + 伴随线
  （可设计 10000λ、NA 0.9 metalens）表明：大面积反设计靠**局域-周期近似（LPA）做超原子级前向 + 角谱传播
  组装全器件**，并以两级优化处理近场耦合残差。这正是把 RCWA 单胞前向规模化到工业孔径的标准路径。
- **改动**：① 新增 `LPAMetalensForward`：RCWA 单胞库（相位/透过率查表，可微插值）+ 角谱传播组装；
  ② 两级流程：LPA 快速全局优化 → 选定区域 RCWA/FDFD 近场耦合修正；③ 与现有 `TwoStageOptimizer`（代理→精算）
  正交组合，形成「LPA 全局 + 代理预筛 + RCWA 精算」三级。
- **DoD**：① ≥ 256×256 单胞孔径在 CPU 分钟级可优化；② LPA vs 全 RCWA 在小孔径上 Strehl 相对误差 < 5%；
  ③ 近场耦合修正使强耦合区 FoM 改善可量化；④ ≥ 8 新测试通过。

### N8.3 — 物理感知潜空间/扩散先验做反设计暖启动（对接 diff-surrogate 生成式底座）🔬
- **动机**：Adv. Optical Materials 14(1), 2026（*Inverse Design in Nanophotonics via Representation
  Learning*）与 OE 33(18):38628, 2025（物理感知潜扩散反设计）表明：用生成式潜空间/扩散先验产出多候选初值，
  再交给可微 EM 精修，能缓解 freeform 反设计的非凸初值依赖。DiffNano 已有 `representation_learning.py`（VAE
  潜空间），但缺「扩散/潜空间采样多候选 → EM 精修选优」闭环。
- **改动**：① 在 `design/` 增加条件式潜采样器（消费 diff-surrogate S8.4 的生成式接口）；② 多候选 → RCWA 批量
  精修 → 用 Strehl/效率打分选优；③ 与 OpenLithoHub 的候选打分器（O8.x）共享 `CandidateScorer` 协议。
- **DoD**：① 潜空间/扩散多候选暖启动收敛优于随机初值（≥ N 个种子的 Wilcoxon 检验）；② 候选选择器接口与
  OpenLithoHub 一致；③ ≥ 6 新测试通过。

### N8.4 — RCWA 后端的不确定度感知 + GPU 基准回填 🖥️
- **动机**：四后端目前只比「梯度/前向一致性」，缺「在哪些 (d/λ, 阶数, 损耗) 工况下哪个后端可信」的
  可量化证据；README 也欠 GPU 基准。配合 §0 主线二与 diff-surrogate S8.1 的共形 UQ。
- **改动**：① 后端选择诊断升级为「工况 → 推荐后端 + 残差带」表；② 在有 CUDA 的机器上回填 RCWA/FDTD GPU
  基准（与 meent 多后端 RCWA 交叉对拍，仅作数值参照，**不 vendoring**）。
- **DoD**：① 后端工况推荐表（含失效边界）；② 与 meent 在 ≥ 1 周期结构上前向相对误差 < 1e-3；③ GPU/CPU
  基准表回填 README（或诚实标注未获 GPU 时给出 CPU-only 结论）。

---

## 2. OpenLithoHub — 计算光刻基准与工作流

**Phase 7 基线**：厚掩模 amplitude/phase 微扰 + Abbe 多源点参照 + `ThickMaskProxy` U-Net；可微形态学
`mrc_projection`（天生 MRC 合规）+ shot-count 估计；`WarmStartProvider` + `CandidateScorer`；
拼接残差 `cross_tile_epe_residual` + overlap×Schwarz 收敛面；已有 EUV `stochastic.py`（随机缺陷/dose）。

### O8.1 — 变分感知（stochastic-aware）ILT：把随机 EPE/LCDU 进优化目标 🔬
- **动机**：arXiv:2602.19027（NVIDIA 生成式 RL ILT，2026-02）在更严的 **EPE 违例阈值（3nm）** 下做后验掩模
  采样并以可微形态学保证曲线 MRC；cuLitho 线（arXiv:2602.15036, 2026）在 pre-silicon 验证里强调
  **through-focus EPE + 随机性**。OpenLithoHub 已有 `stochastic.py` 评测，但 ILT 优化目标仍是确定性
  nominal-image；缺「把随机 EPE/LCDU 期望/分位数直接进 loss」的变分感知优化。
- **改动**：① 把 `stochastic` 指标（LCDU、随机缺陷率、through-focus EPE 分位数）做成**可微/可采样的优化目标项**；
  ② 提供 risk-aware 目标（CVaR/分位数）而非仅均值；③ 与 PV-band 联动给出「随机过程窗口」。
- **DoD**：① stochastic-aware loss 可端到端反传；② 优化后随机 EPE 分位数/LCDU 优于确定性 ILT（回归表）；
  ③ ≥ 10 新测试通过。

### O8.2 — 扩散/后验候选生成器：把 warm-start 从「单初值」升级为「后验掩模族」🔬
- **动机**：生成式 RL ILT 的关键是**后验采样多个掩模**再批量快 ILT 选优（论文 Fig.4 五个后验样本）。
  OpenLithoHub 的 `WarmStartProvider` 目前给单点初值；缺后验样本族 + 批量精修选优闭环。
- **改动**：① 新增 `PosteriorWarmStart`（条件采样多候选，可由轻量扩散/能量模型或 GenAI 接口驱动）；
  ② 批量 ILT 精修 → `CandidateScorer`（EPE+PVB+MRC）选优，复用 Phase 7 的打分器；③ 与 DiffNano N8.3
  共享候选协议。
- **⚠️ 合规**：若引用 NVIDIA cuLitho/生成式 RL ILT 思路，仅从论文 clean-room；公开权重不可用时坦诚标注为
  「结构演示，非 SOTA」。
- **DoD**：① 后验候选族 → 批量精修 → 选优闭环；② 选优结果优于 Phase 7 单初值 warm-start；③ ≥ 8 新测试通过。

### O8.3 — 全芯片拼接的 GPU 多进程化 + 与公开基准交叉对拍 🖥️
- **动机**：2025–2026 综述反复强调「分块引入边界拼接误差、full-chip ILT 需 GPU 加速消除子分区伪影」。
  OpenLithoHub 已有 Schwarz 拼接 + 残差量化 + `inference/multiproc.py`，但缺 GPU 路径与公开基准对拍。
- **改动**：① 把 Schwarz 拼接前向/ILT 做 GPU batch 化（tile 并行）；② 在 LithoBench/ICCAD13 子集上跑端到端
  EPE/PVB/MRC，与公开数字交叉对拍（**仅数值参照**，不 vendoring 第三方代码）。
- **DoD**：① GPU/多进程 tile 并行可用（或 CPU 多进程退化路径）；② ICCAD13 子集端到端指标表 + 与公开结果对比
  讨论；③ 拼接残差随 tile 数的回归曲线。

### O8.4 — `resist_model` / `surrogate_ilt` 的真实化与梯度保真门 🔬
- **动机**：`models/surrogate_ilt.py`、`resist_model.py` 等仍偏占位/无预训练权重。配合 §0 主线一，凡是进
  优化回路的可微代理都应过「代理梯度 vs 高保真有限差分」门（Phase 7 已在四仓各立一处，此处补 litho 侧）。
- **改动**：① 给 resist/ILT 代理补最小可用前向（即便非 SOTA）；② 加梯度保真度回归（代理 vs Born/Abbe 前向
  有限差分）；③ 接 diff-surrogate 的 Sobolev 导数信息训练（S7.1 复用）。
- **DoD**：① 代理可真正产出非随机输出；② 梯度余弦 > 0.99（对高保真前向）；③ ≥ 8 新测试通过。

---

## 3. DiffCFD — 可微流体与逆向设计 / RL

**Phase 7 基线**：多项式/样条 EOS（sCO₂ 可解释对照）；`coupled_fixed_point_gradient` 块伴随 + GMRES 的
强耦合 FSI 隐式微分；`BlowingSuctionEnv` + `ADGradientOptimizer`（AD vs random 对照）；容器化旗舰 sweep +
`cross_validate.py`（Ghia/Poiseuille 解析解）。device-aware 代码存在但 README 标注 CPU-only。

### C8.1 — 旗舰 sweep 的 GPU/JAX 对照 + 与 Diff-FlowFSI 交叉对拍 🖥️🔬
- **动机**：Diff-FlowFSI（CMAME 2025，JAX GPU 可微 CFD/FSI，隐式函数定理穿压力 Poisson + 梯度检查点控长
  rollout 显存）是当前可微 CFD/FSI 的开源参照。DiffCFD 已有强耦合 FSI 隐式微分与解析解对拍，但缺与**第三方
  可微 CFD**在同一算例上的交叉对拍，以及 GPU 实测。
- **改动**：① 在至少一个共有算例（如 FSI 弹性边界 / 顶盖驱动腔）上与 Diff-FlowFSI 做前向 + 梯度交叉对拍
  （仅数值参照，不 vendoring）；② 若有 GPU，回填 SIMPLE 前向/隐式反传的 GPU 基准；③ 把长 rollout 的梯度
  检查点策略文档化（对照 Diff-FlowFSI 的 checkpoint + 隐式混合）。
- **DoD**：① 与 Diff-FlowFSI 在 ≥ 1 算例上前向相对误差 < 2%、梯度方向余弦 > 0.99；② GPU 基准表（或诚实
  CPU-only 标注）；③ 长 rollout 显存策略说明 + 回归。

### C8.2 — RL × 可微边界的混合控制：PPO + AD-梯度 warm-start / 微调 🔬
- **动机**：JFM 2025 可微边界控制 + arXiv:2509.10185（*3D 机翼分离控制 DRL，GPU CFD 数据生成*）共同表明
  「可微梯度做策略热启动/微调」是主动流控提效主线。DiffCFD Phase 7 已有 AD-梯度优化器与 RL 环境，但二者
  是**并列对照**，未做混合。
- **改动**：① 在 `envs/` 用 AD 梯度给 PPO 策略做 warm-start 或梯度增强（actor 预训练 / 混合优势）；② 对照
  纯 PPO、纯 AD、混合三条曲线（同一吹吸控制任务）。
- **DoD**：① 混合控制样本效率优于纯 PPO（学习曲线 + 统计检验）；② AD warm-start 不破坏最终策略性能；
  ③ ≥ 6 新测试通过。

### C8.3 — 瞬态 CHT/FSI 的离散伴随与不稳态梯度 🔬
- **动机**：非稳态 CHT 离散伴随（SU2, 2025）、CODA AD 抽象层 / tape 策略（DLR, AIAA SciTech 2026）指出
  瞬态共轭问题的 tape/伴随是工程级标配。DiffCFD 现有隐式微分面向**稳态**不动点；瞬态 CHT/FSI 仍空白。
- **改动**：① 给 `HeatTransfer2D` / FSI 提供瞬态时间步的离散伴随（checkpoint + 反向 tape）；② 最小不稳态
  共轭算例 + 与有限差分对拍。
- **DoD**：① 瞬态伴随梯度 vs 有限差分相对误差 < 1e-3；② 显存随步数受控（checkpoint 策略）；③ ≥ 5 新测试通过。

### C8.4 — sCO₂ 工作流的校准不确定度（对接 diff-surrogate 共形 UQ）🔬
- **动机**：sCO₂ 在近临界点物性剧变，MLP 代理误差大且**误差大小本身随工况变化**。配合 §0 主线一，给
  `SCO2Surrogate` 配校准覆盖率，而非仅点预测 + 多项式对照。
- **改动**：① 用 diff-surrogate S8.1 的功能/标量共形预测给 sCO₂ 物性预测带覆盖率保证；② 在 PCHE/CHT 工作流里
  把物性不确定度传播到 Nusselt/压降的区间。
- **DoD**：① sCO₂ 物性预测达到目标覆盖率（如 90/95%）；② 工作流输出带不确定度区间；③ ≥ 5 新测试通过。

---

## 4. diff-surrogate — 统一可微代理框架（共享库）

**Phase 7 基线**：`SobolevLoss` + `gradient_fidelity_score`；`PointCloudGeometry`（KNN 多尺度邻域注意力）+
`IrregularMeshEncoder`；`UncertaintyTriggeredSampler` + `MultiFidelityActiveLearner`；Ensemble 产出方差
+ `uncertainty_calibration` 统计位（**但无覆盖率保证**）；DLPack interop + 40 跨仓兼容测试。

### S8.1 — 功能空间共形预测（functional conformal prediction）：校准覆盖率成为一等公民 🔬 ★
- **动机**：Ma/Azizzadenesheli/Anandkumar（*Calibrated UQ for Operator Learning via Conformal Prediction*,
  risk-controlling quantile neural operator，2D Darcy + 3D 车面压力）、IOPscience *ML: Sci. Technol.* 2026
  （CP 给 NN 代理在最高 2000 万维输出上提供有保证的边际覆盖，含 OOD）、arXiv:2509.04623（**函数空间** split-CP）
  共同确立：分布无关、有限样本的覆盖率保证是科学代理的标配。diff-surrogate 现有 Ensemble 方差**不带覆盖率保证**。
- **改动**：① 新增 `conformal.py`：split-CP / risk-controlling quantile 校准，支持标量与函数值输出；
  ② 新增 `coverage_score`（实测覆盖率 vs 目标）与「带宽效率」指标；③ 与 `UncertaintyTriggeredSampler`
  协同（用校准带宽而非原始方差触发采样）。
- **DoD**：① 在 toy Darcy/2D 问题上实测覆盖率达目标（如 90%）且带宽优于裸 Ensemble；② OOD 算例覆盖率不
  灾难性崩塌的诊断；③ 公共 API 导出；④ ≥ 8 新测试通过。

### S8.2 — 结构保持算子：守恒律/有限元外微积分归纳偏置改善 OOD 几何 🔬
- **动机**：Geo-NeW（arXiv:2602.02788, 2026-02）用学到的微分算子 + 兼容的约化有限元空间（有限元外微积分）
  **精确保持守恒律**，在强 OOD 几何上显著超过无约束 transformer，且在分布内也更优（容量更小却更准）。
  diff-surrogate 的 cross-attn/点云算子缺结构保持归纳偏置。
- **改动**：① 在 `geometry/` 或新 `structure.py` 增加散度/通量守恒的投影层（最小可用：离散散度约束）；
  ② 提供「无约束 vs 结构保持」在 OOD 几何（如训练方腔圆障、测试变角台阶）上的对照。
- **DoD**：① 结构保持版在 OOD 几何上误差显著低于无约束基线；② 守恒量残差可量化接近零；③ ≥ 8 新测试通过。

### S8.3 — PDE 预训练 + 迁移：GAOT/Poseidon/DPOT 式可迁移底座雏形 🔬🖥️
- **动机**：GAOT（NeurIPS 2025）明确把自己定位为 PDE 基础模型骨干，延续 Poseidon/DPOT 的「多 PDE 预训练 →
  少样本迁移」。diff-surrogate 已有 cross-attn + 几何编码，但每个下游任务都是从头训；缺**预训练 + 迁移**证据。
- **改动**：① 提供多任务预训练入口（几个 toy PDE 上预训练共享编码器）；② 下游 few-shot 微调 vs 从头训对照；
  ③ 把迁移增益做成回归指标（few-shot 数据效率比）。
- **DoD**：① 预训练→few-shot 微调在 ≥ 2 个下游任务上数据效率优于从头训；② 迁移增益回归化；③ ≥ 6 新测试通过。

### S8.4 — 生成式先验接口：为 DiffNano/OpenLithoHub 的候选生成提供统一底座 🔬
- **动机**：N8.3（光子潜扩散暖启动）与 O8.2（光刻后验掩模族）都需要「条件式多候选生成 → 可微精修选优」。
  应在共享底座提供统一的生成式先验接口，避免两仓各造轮子。
- **改动**：① 新增 `generative.py`：条件采样协议（轻量扩散/VAE/能量模型可插拔）+ 候选打分协议（与 Phase 7
  `CandidateScorer` 对齐）；② DLPack interop 保证跨仓张量零拷贝。
- **DoD**：① DiffNano、OpenLithoHub 各能通过统一接口取候选；② 候选协议跨仓一致性测试；③ ≥ 6 新测试通过。

---

## 5. 跨仓收尾（Phase 8 的统一门控）

- [ ] **校准覆盖率成为统一验收门**：四仓各至少一个核心代理路径补「共形覆盖率 vs 目标」回归（与 Phase 7 的
      梯度保真度门并列）。
- [ ] **GPU 路径 + 开源参照交叉对拍**：DiffNano↔meent(RCWA)、DiffCFD↔Diff-FlowFSI、OpenLithoHub↔LithoBench/
      ICCAD13 各至少一条交叉对拍（仅数值参照，**不 vendoring**，CI 跑 `scancode`）。无 GPU 时坦诚保留 CPU-only 标注。
- [ ] **生成式候选协议跨仓统一**：S8.4 接口被 N8.3、O8.2 复用，候选打分协议一致。
- [ ] **结构保持 + 迁移证据并入 README**：S8.2/S8.3 的 OOD 与 few-shot 增益表回填竞品/参考表。
- [ ] **全仓 CI 全绿**：四仓 + diff-surrogate 联合 CI（含 `scancode` 许可扫描）一次性绿。
- [ ] **honesty boundaries 更新**：把「无 GPU 基准 / toy scale / 无第三方验证」按本期实际进展逐条更新（哪些
      已交叉对拍、哪些仍待 GPU）。

---

## 6. 关键参考文献（Phase 8 新增，2024–2026）

**不确定度 / 校准（主线一）**
- Ma, Azizzadenesheli, Anandkumar, *Calibrated Uncertainty Quantification for Operator Learning via
  Conformal Prediction*（risk-controlling quantile neural operator；2D Darcy + 3D 车面压力），arXiv:2402.01960。
- *Uncertainty quantification of surrogate models using conformal prediction*, Machine Learning: Sci. Technol.,
  2026（CP 在最高 ~2×10⁷ 维输出上保证边际覆盖，含 OOD）。
- *Split Conformal Prediction in the Function Space with Neural Operators*, arXiv:2509.04623（函数空间 split-CP）。

**神经算子 / 几何泛化 / 基础模型（主线三）**
- Geo-NeW, *Structure-Preserving Learning Improves Geometry Generalization in Neural PDEs*, arXiv:2602.02788,
  2026-02（有限元外微积分守恒律归纳偏置，强 OOD 几何）。
- GAOT, NeurIPS 2025, arXiv:2505.18781（PDE 基础模型骨干，迁移学习）。
- DD-DeepONet, Eng. Appl. Artif. Intell., 2026（结构化子域分解，更小 VRAM）。
- Poseidon / DPOT（多 PDE 预训练，迁移）— GAOT 引用的基础模型先例。

**电磁 / 光子反设计**
- *Time Reversal Differentiation of FDTD for Photonic Inverse Design*, ACS Photonics, 2024（98% AD 显存削减）。
  ⚠️ clean-room。
- FDTDX, *High-performance open-source FDTD with automatic differentiation*, J. Open Source Softw. 11:8912, 2026
  （多 GPU 3D AD-FDTD，开源参照）。
- *Two-level optimizer for large-scale metasurfaces with strong near-field coupling*, Optics Express 34(2):1602,
  2026。
- Marzban, Adibi, Pestourie, *Inverse Design in Nanophotonics via Representation Learning*, Adv. Optical Mater.
  14(1), 2026（潜空间/生成式反设计）。
- *Inverse design for scalable photonic systems*, Nature Reviews Materials, 2026-04（大面积反设计综述）。

**计算光刻**
- *Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning*, arXiv:2602.19027, 2026-02
  （后验掩模采样 + 严格 EPE 阈值 + 曲线 MRC）。
- *Transforming Computational Lithography with AC and AI*（cuLitho，pre-silicon MRC/through-focus EPE 验证），
  arXiv:2602.15036, 2026。
- *Full-chip EUV curvilinear mask optimization*, Light: Advanced Manufacturing 7:049, 2026（承接 Phase 7 O7.1）。
- LithoBench / ICCAD13（ILT 公开基准，交叉对拍用）。

**可微 CFD / FSI / 流控**
- Diff-FlowFSI, *GPU-optimized differentiable CFD/FSI*, CMAME, 2025（JAX GPU，隐式函数定理 + 梯度检查点，
  开源参照）。
- *Discovering Flow Separation Control Strategies in 3D Wings via Deep RL*, arXiv:2509.10185, 2025（GPU CFD +
  DRL，工业级几何）。
- 非稳态 CHT 离散伴随（SU2, 2025）；CODA AD 抽象层 / tape 策略（DLR, AIAA SciTech 2026）。

---

### 优先级建议（若资源受限，先做这 5 条）
1. **S8.1 功能空间共形预测** ★ — 一处改动、四仓共享底座受益，直接兑现 Phase 8「不确定度可信」主线，且与
   Phase 7「梯度可信」门对称收口。
2. **N8.1 FDTD 时间反演伴随** — 有 2024 论文背书的 EM 专属显存质变（98%），是 DiffNano 走出 toy 网格的关键。
3. **O8.1 变分感知 ILT** — 复用既有 `stochastic.py`，把随机 EPE/LCDU 进 loss，对接 2026 cuLitho/生成式 RL 主线。
4. **S8.2 结构保持算子** — 低成本归纳偏置，直接改善 OOD 几何泛化（Geo-NeW 2026-02 背书）。
5. **跨仓 §5「GPU + 开源参照交叉对拍」** — 把四仓从「自测玩具」推向「外部可复算 + 有横向参照」，是 Phase 8
   收尾的工程门控。

> **资源前置依赖**：N8.1/N8.4/C8.1/O8.3/S8.3 标 🖥️ 的任务需要至少一台 CUDA GPU；若暂无 GPU，可先完成其
> CPU 正确性与接口（time-reversal 伴随、LPA 前向、共形校准、结构保持层都可在 CPU 验证数值正确性），把 GPU
> 基准回填留到资源到位，并在 README 继续诚实标注。
