<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnMtpDecode 开发任务拆分

设计文档：
`third_party/xllm_ops/xllm_ops/mega_gdn_mtp_decode/design.md`。

## 1. 里程碑与依赖

```text
M0 设计/Golden
 ├─ TL1 TileLang 语义原型
 │    ↓
 │  TL2 PTO lowering + source audit
 │    ↓
 │  TL3 六种 K AOT object
 │    ↓
 │  M2 生产 PTO kernel
 ├─ M1 Host ABI + tiling
 └─ M3 xLLM metadata contract
       ↓
M4 ACLNN + eager integration
       ↓
M5 Graph + Prefix Cache
       ↓
M6 精度闭环
       ↓
M7 msprof + 模型 A/B
```

每个任务只能在依赖项通过后进入下一阶段。性能修改必须以已通过精度的
版本为基线。

### 当前状态（2026-08-02）

| 范围 | 状态 | 证据 |
| --- | --- | --- |
| Python Golden | 完成 | K=1～16 共 48 cases passed |
| TileLang PrimFunc/source/AOT | 功能与纯 DSL 调优完成，非生产 runtime | 25 项 host/source audit；六种 K lowering/AOT；K1～K5 保持基线、K8 使用 norm-weight 与 Q/K native cache |
| TileLang direct-library NPU 数值 | 完成 | 六 K 34/34 四输出逐 bit；K8 最终特化跨两张卡新增 B1 24/24 与 B4 smoke；half-sentinel 80/80；Softplus/ownership 扩展门禁通过 |
| 生产 PTO/OPP | K1～16、B1/4/8、Qwen3.5 head 几何工程性能门禁完成 | 原 key + K10～16 deferred-Norm key 210～216；960/960 点相对 seq17 小算子链至少提升 10%，最弱 11.945% |
| 生产 ACLNN NPU Golden | K1～16 与 Qwen3.5 几何设备门禁完成 | 扩展 suite 832/832：含四模型组 × 5 TP × 16 K × same/fork 的 640 项 Qwen3.5 head 几何 |
| 纯 TileLang DSL 单变量调优 | 完成 | readout cache、K8 norm-weight cache、K8 Q/K native cache 已保留；小 K 保持逐字节基线 |
| 分段 TileLang owner 调优 | 设备验收完成，暂不建议生产启用 | 正式 AOT 四输出 36/36、重复启动 256 轮通过；六 K fresh ABBA 的 eager 整链快于小算子链，但 profiler 纯 kernel 和 ACLNN 对比未过性能门禁 |
| 生产 PTO 单变量调优 | 第八轮 deferred-Norm 接受，全工程矩阵目标完成 | 最重 NV64/B8/TP1 ABBA 最弱 K16 +12.80%；全 960 点最弱 +11.945%；object `f56a00...197` |
| xLLM eager 路由 | 父仓编译与专项 UT 完成，待模型 smoke | Release 目标构建通过；wrapper 3/3、batch 41/41、ACL Graph 11/11，共 55/55；dense verify 可达 |
| Graph/Prefix Cache owner | 未完成 | Graph 暂时禁用融合；P1～P3 ownership/lease 待闭环 |
| Qwen3.5 算子几何泛化 | 完成 | 八个官方模型归并为 `NK=16, NV=16/32/48/64`，按 TP1/2/4/8/16 形成 20 种 rank-local 几何；K1～16 × same/fork 共 640/640 通过 |
| 模型精度/性能 | 未完成 | 27B 本地权重可用于 V2～V4；其余模型未在本机完成端到端验收 |
| 版本管理 | 未完成 | 新算子目录和父仓接入文件未跟踪；发布前需先提交子模块再更新 gitlink |

## 2. TileLang 原型与 PTO 生成

### T1：TileLang 语义原型

- Owner：算子开发
- 依赖：M0、Golden G1/G2
- 产出：
  - `mega_gdn_mtp_decode.py`；
  - 共用的 `build_mega_gdn_mtp_decode_kernel()`；
  - K=1/2/3/4/5/8 specialization。
- 验收：
  - 六种 K 均能构造 PrimFunc，K=6/7 fail-closed；
  - Conv、Q/K L2Norm、gate、recurrent、checkpoint、RMSNorm/Z gate 均在
    同一 DSL 中；
  - BF16→FP32→BF16 舍入点与 Golden 一致；
  - read/write slot 和 accepted checkpoint 地址与设计文档一致。
- 估时：1～1.5 人日

### T2：PTO lowering/JIT adapter 同源路径

- Owner：算子开发
- 依赖：T1
- 产出：JIT/AOT 共用的 PTO lower + source transform。
- 验收：
  - JIT 编译的是插入 vector guard 和 full-mask 初始化后的 source；单体
    AOT 另外保留 mixed task type，分段 AOT 不覆盖正式 runtime task
    metadata；
  - 不再直接调用未经 transform 的 `@tilelang.jit(target="pto")`；
  - K=1/2/3/4/5/8 的 PrimFunc、lowering、ABI、UB 和 source audit 通过；
  - TileLang 0.1.4/CANN 9.0.0 使用受控 direct-library harness 执行四输出
    NPU 门禁；生产 runtime/ACLNN 仍由 K6 单独验收。
- 估时：1～2 人日

### T3：generated PTO source 审计

- Owner：算子开发
- 依赖：T1
- 验收：
  - TileLang PrimFunc 的 18 个业务参数顺序固定；
  - generated exported `call()` 为上述 18 个参数加 `stream`，共 19 个；
  - vector body guard、`set_mask_norm()` 和
    `set_vector_mask(-1, -1)` 各出现一次；
  - 单体 source 的 mixed task type marker 出现一次，三个分段 source
    均不自行覆盖 task type；
  - 六种 K 的 `TROWEXPAND/TCOLSUM/TCOLEXPAND` 计数符合时间循环 lowering；
  - 最大 UB offset 加 tile 大小小于 192 KiB；
  - Conv/SSM checkpoint 地址公式与生产 PTO 一致；
  - source pattern 变化时 fail-closed。
- 估时：0.5～1 人日

### T4：六种 K 的 PTO AOT

- Owner：算子开发
- 依赖：T2、T3
- 验收：
  - 每种 K 都生成 `.cpp/.o`、registry、manifest；
  - manifest target 为 `pto`；
  - fingerprint 包含 `dav-c220`、`MEMORY_BASE` 和 PTO include；
  - fingerprint 还包含参与构建的 PTO header 内容哈希；
  - 六个 variant ABI 完全一致，object 非空。
- 估时：0.5～1 人日

当前 `.cpp/.o`、registry 和 manifest 已生成。AOT fingerprint 已覆盖实际
compile flags/include、Bisheng resolved path/version/SHA256、TileLang/PTO
header tree SHA256；header tree 在 family 级只计算一次。A2/A5 默认全量构建
跳过只支持 A3 的 PTO family，显式请求仍 fail-closed。JIT/AOT 均优先使用
经过校验的 TileLang bundled PTO header。

六个 `.o` 是 K=1/2/3/4/5/8 的静态 specialization，不是一次调用下发六个
Kernel。对外仍是一个 `MegaGdnMtpDecode` family，runtime 按 K 只 dispatch
一个 entry，并且每层只 launch 一次。隔离测试目录中的 `k*.so` 不进入生产
打包；生产形态应把六个 object、一个 `registry.inc` 和一个 wrapper 链入同一
`tilelang_kernels` target。当前 CMake 尚未注册该 family，接入前不得把测试
`.so` 当成生产交付件。

TileLang 原型当前采用 `batch × key-head` owner：一个 AIV 负责对应的 Q/K
Conv shard、所辖 value-head Conv/recurrent/checkpoint/norm，producer 与
consumer 保持在同一 AIV，以规避 TileLang PTO 0.1.4 缺少跨 AIV
`T.sync_all` lowering。默认 B4/NK8 时 launch 16 个 mixed blocks。

### T5：纯 TileLang DSL 单变量性能调优

- Owner：算子开发
- 依赖：T1～T4
- 当前接受源码 SHA256：
  `c6ec0cd0cbcb27932bc418717e473928d9ba6429db5258293c5af6df391780b0`。
- 已保留：
  - readout 保持在 UB，K8/B1 fresh ABBA p50
    `1595.057 -> 1537.914 us`，下降 3.583%；
  - `norm_weight` 仅在 K8 task scope 搬入并转换一次。K8/B1 fresh ABBA
    balanced p50 `1543.032 -> 1538.640 us`，下降 0.285%；
  - Q/K cache 仅在 K8 使用 consumer-native 三维 UB 布局，直接
    `TROWEXPAND`，删除两次 cache-to-column `TMOV`。K8/B1 fresh ABBA
    balanced p50 `1553.394 -> 1546.311 us`，下降 0.456%，加速
    1.00458x；四次运行均无 outlier；
  - K8 UB 高水位由 168,800 B 降至 168,288 B；
  - K1～K5 generated C++ 和 `.so` 与上一接受基线逐字节相同。
- 已拒绝：
  - Owner-private Conv BF16 cache：首个 fresh paired p50
    `1543.007 -> 1549.825 us`，回退 0.442%，其余运行未满足完整有效
    ABBA 门禁；
  - Conv weight `[4,128]` 合并 load/cast：精度通过，但 fresh ABBA
    只有一组改善，balanced p50 仅约 0.083%；
  - checkpoint state-store 4 路展开：精度通过，但 K8/B1 balanced p50
    `1545.949 -> 1552.143 us`，回退 0.401%；
  - 对所有 K 无条件外提 `norm_weight`：K1 balanced p50 回退约 0.203%，
    因此只保留 K8 specialization。
  - 对所有 K 无条件使用 Q/K consumer-native cache：K1 两组分别
    `415.489 -> 416.040 us` 和 `415.474 -> 415.478 us`，因此只保留
    K8 specialization。
- 精度门禁：
  - 六 K、same-slot/prefix-fork、first/mid/last accepted checkpoint 的
    四输出逐 bit；
  - 最终 K8 specialization 在两张健康空闲卡新增 24/24 逐 bit 通过；
  - 最终 K8/B4 prefix-fork midpoint smoke 四输出通过；
  - half-sentinel 80/80；
  - Softplus 阈值、负无穷与 state ownership 扩展用例。
- 生产 AOT 门禁：
  - K=1/2/3/4/5/8 六个 `.cpp/.o`、registry 和 manifest 一次构建通过；
  - manifest 包含六个 BF16 specialization，target 为 `pto`。
- 约束：本阶段只改 TileLang DSL；checkpoint barrier/event、手改 generated
  PTO source 和生产 PTO kernel 调优全部延期。

TileLang 优化指导书检查项已闭环：

| 检查项 | 结论 |
| --- | --- |
| 算法与融合 | Conv、Q/K L2Norm、gate、recurrent、checkpoint、RMSNorm/Z gate 已在单一 DSL 中 |
| Tile 与任务映射 | `batch × key-head` owner；B4 映射 32 个 AIV task，无多波次 Persistent 收益条件 |
| 数据复用 | state、gate、readout 常驻 UB；K8 额外缓存 norm weight 和 consumer-native Q/K |
| 核内流水 | recurrent token 间存在严格 state 依赖；K8 UB 仅余 28,320 B，小于另一份 32,768 B state half |
| C/V 流水 | 无独立 Cube producer/consumer stage，不应用 `T.Pipelined` C/V 模式 |
| Vector/指令合并 | Broadcast、reduce、`mul_add_dst` 和 vector gate 已使用；额外打包/展开候选实测回退 |
| Persistent | 逻辑 owner 数不超过目标 Vector Core 数，不引入常驻调度循环 |
| Autotune | head/tile 受 128×64 state half 和 192 KiB UB 约束；采用六种 K 静态 specialization 与 fail-closed fallback |
| 生成 PTO 事件 | checkpoint barrier grouping 属于下一阶段 PTO 调优，不混入纯 TileLang DSL 阶段 |

### T6：分段 TileLang owner 调优

- Owner：算子开发
- 依赖：T1～T5、`MegaGdnDecode` owner map
- 当前结构：

  ```text
  Conv：          40 channel owners，owner 内循环 batch
    ↓ launch boundary / GM visibility
  Recurrent：     B × 24 value heads × 2 state halves
    ↓ launch boundary / GM visibility
  RMSNorm/Z gate：B × 24 value heads
  ```

- 为什么是三段：
  - Conv 与 Recurrent 的最佳 owner 分别是 channel 和 value-head；
  - TileLang 0.1.4 无法 lower 跨 AIV `T.sync_all`，用 launch boundary
    完成 ownership hand-off；
  - 完整 `128×128` FP32 state 加两个完整 compute/broadcast Tile 超过
    192 KiB UB；
  - 动态 half slice 不能直接作为 Tile op 的 `access_ptr`，因此
    Recurrent 暂按两个 `128×64` half owner，Norm 单独汇合。
- AOT 形态：
  - 三个 family：
    `mega_gdn_mtp_decode_{conv,recurrent,norm}`；
  - 每个 family 支持 K=1/2/3/4/5/8，共 18 个静态 object；
  - runtime 每次只按当前 K 下发 3 个 kernel，不是 18 个。
- 已完成门禁：
  - 六 K 的三段 PrimFunc/lowering；
  - 三个 registry family，各 6 个 PTO specialization；
  - 三段 ABI、owner 数、UB 高水位和 source pattern；
  - Conv 改为 40 个 channel owner 内循环 batch，避免按 B 重复搬四组
    weight；
  - 三个 family 已链入正式 `tilelang_kernels`，wrapper 按当前 K 各查找
    一个 entry，固定顺序下发 Conv、Recurrent、Norm；
  - K=1/2/3/4/5/8 共 18 个 AOT object 编译成功，wrapper 增量构建和
    CPU 输入拒绝用例通过；
  - 三段 generated Vector body 在任何 TCVT/Vector 指令前显式恢复
    full mask，避免继承前序 kernel 的 partial/count mask。
- 设备验收（Ascend 910B3，CANN 9.0.0，B=1）：
  - 正式 AOT 四输出 Golden 36/36 通过，覆盖六 K、accepted
    first/mid/last、same-slot/prefix-fork；
  - K1 同进程 256 次重复启动并穿插 Vector mask 污染后通过，未再出现
    `507015`；
  - 六 K fresh-process ABBA 的三段整链 NPU Event 均快于当前小算子链，
    平均延迟降低 47.69%～86.59%；
  - profiler 的纯 kernel duration 未过门禁：K1/K8 三段分别为
    157.50/534.39 us，小算子 kernel 和为 95.15/159.62 us，ACLNN 单
    kernel 为 18.99/46.70 us；
  - 因此设备验收已完成，但分段 TileLang 只证明 eager 下发融合收益，
    不满足生产最优 kernel 的启用条件；当前保持 opt-in/fail-closed。
- 停止条件：
  - 若正式 AOT 精度失败，先修语义，不做性能；
  - 若三段仍显著慢于小算子且 Profile 指向 128 行 checkpoint store，
    记录 TileLang 表达上限后再进入 PTO 二维 burst copy；不得提前混调。

## 3. Kernel 与 OPP

### K1：op definition 与 shape inference

- Owner：算子开发
- 依赖：M0
- 产出：
  - `mega_gdn_mtp_decode_def.cpp`
  - `mega_gdn_mtp_decode_proto.cpp`
  - Host CMake
- 验收：
  - 13 输入、4 输出顺序与 ACLNN schema 一致；
  - BF16/FP32/INT32 dtype 正确；
  - 四个输出 shape 推导正确；
  - 非法 K、state length、checkpoint stride 返回失败。
- 估时：0.5～1 人日

### K2：Host tiling 与模板 key

- Owner：算子开发
- 依赖：K1
- 产出：
  - `mega_gdn_mtp_decode_tiling.h/.cpp`
  - K=1/2/3/4/5/8 的 tiling key；
  - K=6/7/9～16 共用的动态 key 100 fallback；
  - UB≥182,816B 时，K=10～16 使用 deferred-Norm key 210～216。
- 验收：
  - 通过平台 API 获取 AIC/AIV/UB/workspace；
  - `block_dim=max(C/128,B*NV)` 并受硬件 core 数限制；
  - tiling 校验 accepted/read/write tensor 的 dtype、rank 和 shape；
  - accepted 值域、slot 值域、batch 内 write 唯一性和跨行重叠由
    xLLM host 元数据校验，tiling 不触发 D2H；
  - schedule mode 为 1。
- 估时：1～1.5 人日

### K3：MTP CausalConv PTO 阶段

- Owner：算子开发
- 依赖：K2、T3、Golden G1
- 产出：PTO Conv channel-owner 代码。
- 验收：
  - 从 read slot 的 `accepted-1` 读取三行；
  - 连续计算 S 个 token；
  - `conv_out` BF16 bitwise；
  - write slot 布局为 `[selected_tail_2, qkv_0..qkv_K]`；
  - read/write 不同时 shared slot bitwise 不变。
- 估时：1.5～2.5 人日

### K4：Recurrent + checkpoint PTO 阶段

- Owner：算子开发
- 依赖：K2、T3、Golden G2
- 产出：按 `batch × value_heads` 的 recurrent owner。
- 验收：
  - 初始 state 每 head 只加载一次；
  - UB 内连续 S 次 update；
  - S 个 checkpoint 全部写入 write slot；
  - `TROWEXPANDMUL + ColSum128` 两次/token；
  - OuterProduct 不改变 FP32 更新顺序。
- 估时：2～3 人日

### K5：RMSNorm/Z gate 与流水

- Owner：算子开发
- 依赖：K3、K4
- 验收：
  - readout BF16 RINT 后再进入 Norm；
  - recurrent 与 Norm 共用 head owner；
  - MTE2/Vector/Scalar/MTE3 event 完整；
  - Conv→recurrent 只有必要的阶段同步；
  - UB 高水位不超过设计值。
- 估时：1～1.5 人日

### K6：OPP/ACLNN 构建

- Owner：算子开发
- 依赖：T4、K1～K5
- 验收：
  - 静态 K=1/2/3/4/5/8、动态 key 100、热点 key 208 和 deferred-Norm
    key 210～216 均生成 object 与 metadata，最终共 15 个 key；
  - ABI 符号和 schema 一致；
  - ACLNN NPU 先通过 state=1 隔离，再通过 K=1～16 四输出 Golden；
  - accepted=1/middle/S、read==write/read!=write、state in-place 均覆盖；
  - Qwen3.5 八个官方模型的 rank-local head 几何覆盖 TP1/2/4/8/16；
  - Real ATK 覆盖 B=1～4；
  - source pattern 检查 fail-closed。
- 估时：1～1.5 人日

### K7：生产 PTO 单变量性能优化

- Owner：算子开发
- 依赖：T5 完成、K6、V3 baseline
- 方法依据：
  - 《PTO-ISA 优化手段指导书》；
  - 《TileLang 算子性能优化指导书》。
- 验收：
  - 先按 K=1/2/3/4/5/8、B=1/4/8 记录 Vector/MTE2/MTE3/Scalar
    时间线、UB 高水位和 checkpoint 写带宽；
  - 每轮只改变一个主要变量，优先验证 Conv token load/store ping-pong；
  - ready/free 依赖成对，稳态无新增 `PIPE_ALL`；
  - recurrent state 在 S 步内只加载一次，不以 GM 重读换取流水；
  - 每个候选先跑四输出 Golden，再跑相同 warmup/measured 的 fresh ABBA；
  - 无稳定收益的 K/B bucket 保留当前单 Buffer fallback；
  - 报告延迟下降率和加速比，不把 Simulator/CostModel 当硬件实测。
- 估时：2～4 人日

第一轮 recurrent-prefetch 结果：

- 测量口径固定为每个 K 独立进程、5000 warmup、50 samples、
  20 launches/sample、fresh ABBA；
- 仅 K8 保留 recurrent 下一 token MTE2 预取，p50 从
  `51.306/51.185 us` 降至 `50.065/49.930 us`，fresh final 包复核为
  `50.158 us`；
- K1/2/3/4/5 和动态 K 的 AIC/AIV 机器码与 serial baseline 一致；
- token unroll 因约 9.9% 回退拒绝；
- checkpoint store/Norm overlap 因 53 项 ACLNN 中 52 项 final out
  精度失败拒绝；
- 第一轮 fresh OPP 通过 39 项 Golden/source audit 和 53 项 ACLNN NPU
  Golden。

第七轮 720ah 结果：

- `K8/B4/NK8/NV24` 保留九行 BF16 readout/Z，在 head-local UB 内延后
  exact Norm，并批处理九行 Q/K L2Norm；
- 正式无 profiling A-B-B-A 两组收益 10.676%/10.048%，平均
  `114.051→102.232 us`，下降 10.363%；
- profile 仅用于因果解释：Task -9.418%、AIV -9.015%、Vector -8.585%、
  MTE3 -12.762%；
- 最终 OPP 只包含 8 个 key：100/101/102/103/104/105/108/208；
- K=1～16 的 address/UB/dispatch 22 项、CPU reference 48 项、设备 ACLNN
  301 项全部通过；设备矩阵包含 120 项 Qwen3.5 全模型族 rank-local head
  几何；
- 逐函数机器码审计仅 key 208 AIV 改变，其他 15 个 AIC/AIV 函数逐字一致。

第八轮 deferred-Norm 结果：

- K10～16 在 UB≥182,816B 时选择 key 210～216，低 UB 平台回退 key 100；
- K16 key 216 保持原机器码，只新增 key 210～215；key
  100/101/102/103/104/105/108/208/216 的 18 个 AIC/AIV 函数逐字不变；
- 地址/tiling 22/22、定向 K10～16 ACLNN 74/74、原完整 312/312、扩展
  Qwen3.5 suite 832/832 均通过；
- 最重 `NV64/B8/TP1` 工程 A-B-B-A 的最弱点 K16 提升 12.80%；
- 四模型组 × TP1/2/4/8/16 × B1/4/8 × K1～16 的 960/960 点工程矩阵
  均提升至少 10%，最弱 `122B/397B、TP1、B8、K16` 为
  `1098.127→966.959 us`，提升 11.945%；
- 最终 object SHA256 为
  `f56a00f1381cdf6bc15025e422004c3adaecb2900c832cd7116ef6e5d25fb197`。

上述 960 点结论为 NPU Event 工程 `device_microbenchmark`。公平性采集器无法
解析本机 NPU 4 mapping，不能升级为正式发布 benchmark；Qwen 模型 MTP TPOT
仍需 V2～V4。

第二轮 K8 Conv BF16 input/output ping-pong 结果：

- 仅修改 K8 AIV；K1/2/3/4/5/动态路径的 AIC/AIV 与 K8 AIC 的逐函数
  SHA256 均不变；
- K8 AIV 从 `5852 B` 增至 `5936 B`，candidate object SHA256 为
  `8b65f1cbcf77eec63ec2d95175e0db92d4d43d48c2b2d1b2155c9c5e6683d927`；
- fresh-process ABBA 继续使用 5000 warmup、50 samples、
  20 launches/sample，基线为第一轮 recurrent-prefetch：

| B | prefetch p50 A1/A2 | ping-pong p50 B1/B2 | 两组配对收益 |
| ---: | ---: | ---: | ---: |
| 1 | 51.273/50.014 us | 48.558/49.139 us | 5.30% / 1.75% |
| 4 | 126.759/126.737 us | 119.540/119.799 us | 5.70% / 5.47% |
| 8 | 208.260/207.422 us | 190.954/190.367 us | 8.31% / 8.22% |

- 同卡 `PipeUtilization` 显示 B4 Task/AIV 分别下降 5.58%/6.73%，B8
  分别下降 8.49%/9.00%；Vector 时间基本不变，B8 MTE2 时间下降 7.61%，
  与 Conv 搬运/写回和 Vector 重叠的假设一致；
- fresh candidate 通过 39 项 Golden/source audit 和完整 57 项 ACLNN
  NPU Golden，第二轮优化保留。

第三轮先修复 SSM 元素地址在大 slot 下的 int32 溢出。`checkpoint`、
`checkpoint_stride` 和 head offset 的独立上界均小于 int32，但
`checkpoint * checkpoint_stride` 最大约 9.66G，因此只在乘法前将
checkpoint 提升为 int64：

- 全变量 int64 版本虽通过 57 项 ACLNN，但相对第二轮 ping-pong 在 B1
  稳定回退 3.44%/3.55%，拒绝；
- 最小 int64 版本通过当时 48 项 Golden/source/address 和 57 项 ACLNN，相对旧
  int32 版本 B1 为 +1.84%/+3.60%，B4 为 +1.09%/-0.43%，B8 为
  +0.27%/+0.50%，保留；
- Q/K L2Norm 的 Vector→Scalar 合并在 B1 出现 -30.86%/-1.85%，拒绝；
- K8 Conv FP32 history 地址 ring 相对最小 int64 在 B1 为
  -0.73%/-2.30%，B4/B8 接近噪声，拒绝。

第四轮仅把 K8 的 Q/K 两次 BF16 row load 合并为一次 `[2,128]` dynamic
stride TLOAD，不改变 TCVT、L2Norm、event 或数值顺序：

| B | minimal-int64 p50 A1/A2 | Q/K pair-load p50 B1/B2 | 两组配对收益 |
| ---: | ---: | ---: | ---: |
| 1 | 48.533/50.170 us | 48.580/49.441 us | -0.10% / 1.45% |
| 4 | 119.688/119.527 us | 118.417/118.147 us | 1.06% / 1.15% |
| 8 | 191.460/191.983 us | 191.056/191.812 us | 0.21% / 0.09% |

- 仅 K8 AIV 改变，机器码从 `5936 B` 降至 `5880 B`；其他六个 AIV 路径
  和全部 AIC 逐函数 SHA256 不变；
- candidate object SHA256 为
  `a5ed03e56582e00f68d9748c628c658b34c81fc8201f6b81a47cc5547b765457`；
- fresh candidate 通过当时 48 项 Golden/source/address 和完整 57 项 ACLNN
  NPU Golden，第四轮优化保留；补充 slot 边界、跨行冲突和 production UB
  footprint 联动后，当前静态/CPU 门禁为 54 项。

第五轮为 `K8/B4/NK8/NV24` 增加独立 tiling key 208：

- 一个 `batch × key-head` AIV owner 归一化一次 Q/K，并在 UB 中复用给
  同组 3 个 value head；其他 shape 继续使用旧 key；
- UB cache 为 `[174112,182304)`，Host reserve 183,552B，低于 A2/A3
  PTO scratch 起点 188,416B；
- candidate object SHA256 为
  `aa0cee40fb222a8f087898d6cfd92df1f00cf3c208e071c973cb717570e84cac`；
  旧 key 100/101/102/103/104/105/108 的 14 个 AIC/AIV 函数均逐字一致；
- 57 项 ACLNN 全部通过；14 records × 4 outputs 与 production bitwise
  一致，same-slot/prefix-fork 均确定；
- K8/B4 fresh A-B-B-A 两组 p50 下降 3.63%/3.74%，中心值
  `118.794→114.421 us`，下降 3.68%；
- K8/B1 fallback 无回退，K8/B8 fallback 仅 +0.13%；
- profile 最后 120 条显示 Task -4.70%、AIV Vector -4.37%、MTE2
  -16.68%，因此第五轮接受。

第六轮只移动 checkpoint MTE3→V free wait 到 token-end drain，以 key 208
为基线，没有修改 checkpoint store 布局、Q/K cache 或 TileLang：

- 机器码隔离通过，仅 key 208 AIV 从 `6552 B` 变为 `6540 B`，其余 15 个
  函数逐字一致；
- 57 项 ACLNN 全部通过，但 14-record exact matrix 中 21/56 个输出
  fingerprint 与 key 208 不同，36 个 output/record CPU gate 失败；
- seed 20263217 的 same-slot `ssm_state` 三次 SHA256 均不同，
  same-slot/prefix-fork final out 也非确定，确认是 UB state source 生命周期
  竞态；
- 候选在性能 A/B 前拒绝，立即 EVENT_ID2 wait 已恢复，PTO source hash
  回到 key208-r1；
- 当时 best 114.421 us 距 114 us 仅 0.421 us，低于 0.500 us 噪声线，
  因此在 key208-r1 checkpoint 暂停；后续第七轮 720ah 以新的 UB
  生命周期假设重新开启并达到 10.363%。

## 4. Golden 与测试

### G1：Conv state Golden

- Owner：测试/算子开发
- 产出：纯 PyTorch/CPU 参考实现。
- 覆盖：
  - K=1～16；
  - accepted=1/middle/S；
  - read==write、read!=write；
  - 非零和共享 read slot。
- 估时：0.5～1 人日

### G2：完整 GDN Golden

- Owner：测试/算子开发
- 依赖：G1
- 验收：
  - 固定 BF16 舍入点；
  - 比较四层输出；
  - 连续 256 轮；
  - 至少 30 个精度 case。
- 估时：1～1.5 人日

### G3：错误合同测试

- 覆盖：
  - K=0、K>16；
  - accepted=0、accepted>S；
  - state length/stride 不匹配；
  - write slot 重复；
  - dtype、rank、contiguous 错误。
- 估时：0.5 人日

## 5. xLLM eager 接入

### E1：ACLNN wrapper 与 API

- Owner：引擎开发
- 依赖：K6
- 产出：
  - `npu_mega_gdn_mtp_decode.cpp`
  - `npu_ops_api.h`
  - NPU CMake
- 验收：
  - state 原地 alias；
  - 输出 shape `[B,S,NV,128]`；
  - 输入错误在 Host 侧明确失败；
  - 不修改 `MegaGdnDecode`。
- 估时：1 人日

### E2：Qwen3.5 spec verify 路由

- Owner：引擎开发
- 依赖：E1、G2
- 验收：
  - 只在 `is_spec_verify` 且 `1 <= K <= 16` 命中新算子；
  - checkpoint stride 为 S；
  - 环境变量可完整回退；
  - 新路径命中时不再执行 split Conv/gating/recurrent/norm。
- 估时：1～1.5 人日

### E3：read/write state metadata

- Owner：引擎开发
- 依赖：M0
- 产出：
  - `ModelEmbeddingInput` read/write ids 与 tensors；
  - batch builder / MTP worker 填充逻辑；
  - 旧 `linear_state_ids` 的同槽兼容规则。
- 验收：
  - metadata 始终 sequence-scoped；
  - `.to(device)` 保持 INT32 contiguous；
  - Host IDs/accepted 覆盖不一致的旧 device tensor；
  - same-slot read/write 与 legacy tensor alias，不重复传输；
  - forked state 在融合未命中时 fail-closed，不进入旧 fallback；
  - CP/DP padding 不打乱映射。
- 估时：1～1.5 人日

## 6. Graph 与 Prefix Cache

### P1：ACL Graph persistent tensors

- Owner：引擎开发
- 依赖：E3
- 验收：
  - read/write/accepted 三个 tensor capture/replay 均更新；
  - 每个 padding 行使用 graph 专属且互不冲突的 sink write slot；
  - sink slot 不与真实请求或 shared-prefix read slot 重叠；
  - A→B capture、C→D replay 只更新 D。
- 估时：1～1.5 人日

### P2：Prefix Cache state lease

- Owner：缓存/调度开发
- 依赖：E3
- 验收：
  - shared slot 只读 lease；
  - private slot 独占写 lease；
  - 首轮 read shared/write private；
  - 后续轮 read/write private；
  - 请求释放、驱逐和异常回滚无泄漏。
- 估时：2～3 人日

### P3：移除 Prefix Cache H2H state copy

- Owner：缓存/引擎开发
- 依赖：P2、E2
- 验收：
  - trace 中无旧 H2H copy/event；
  - shared state bitwise 不变；
  - eager 与 Graph 都命中新 ABI。
- 估时：0.5～1 人日

## 7. 验证与发布

### V1：ATK 和图模式

- K=1～16 × B=1～4；
- 同槽与跨槽；
- capture/replay 动态 state id；
- 失败即阻断性能测试。
- 估时：1～1.5 人日

### V2：模型精度

- 固定模型、权重、TP 和 sampling；
- old/new MTP acceptance 完全一致；
- token、accepted count、任务集指标 A/B；
- Prefix Cache on/off。
- 估时：1.5～2 人日

### V3：纯 kernel msprof

- split baseline 与新算子 fresh ABBA；
- 每个 K、B 独立报告；
- 分离 checkpoint 写带宽；
- 保存 source/object hash 与原始 CSV。
- 2026-08-02 工程验收：最重 NV64/B8/TP1 A-B-B-A 全 K 超过 10%；Qwen3.5
  全几何 960/960 点相对 seq17 小算子链超过 10%，最弱 11.945%。因公平性
  snapshot 解析失败，正式发布 benchmark 仍未完成。
- 估时：1～1.5 人日

### V4：模型 MTP TPOT

- eager/Graph、Prefix Cache on/off；
- 并发 1/4；
- route gate：新 kernel 次数符合层数×verify step，旧路径为 0；
- 关闭 profiler 后做正式 TPOT。
- 估时：1～1.5 人日

### V5：版本与构建可复现性

- 先在 `third_party/xllm_ops` 中跟踪并提交
  `xllm_ops/mega_gdn_mtp_decode/`，再更新父仓 submodule gitlink；
- 父仓任务文档、测试、TileLang kernel、ACLNN wrapper 与引擎路由全部纳入
  同一变更集；
- AOT fingerprint 覆盖 flags、include 路径和 PTO header 内容，不依赖
  `--force` 才能拾取源码变化；
- 发布 artifact 记录父仓/子模块 commit、OPP object SHA256 和测试日志。
- 2026-07-31 已使用现有
  `build/cmake.linux-aarch64-cpython-311` Release tree 构建
  `batch_test`、`mega_gdn_mtp_decode_wrapper_test` 和
  `acl_graph_executor_test`；三组 C++ UT 分别 41/41、3/3、11/11，
  共 55/55 通过。手工 CMake 流程必须显式设置 CANN/ATB、Python、
  PyTorch/PyTorch-NPU 路径，并将 PyTorch wheel 的 `torch.libs` 加入
  `LD_LIBRARY_PATH`。
- 估时：0.5～1 人日

## 8. 建议排期

两人并行时：

| 周期 | 算子开发 | 引擎开发 |
| --- | --- | --- |
| 第 1 周 | G1/G2、T1～T3、K1/K2 | E3、P1、Prefix owner 设计 |
| 第 2 周 | T4、K3～K6、ATK 修复 | E1/E2、P2/P3、Graph |
| 第 3 周 | msprof 与 kernel 优化 | 模型精度/TPOT、发布门禁 |

完整生产版本仍按 17～27 人日估算。POC 不得用未执行的 NPU 测试抵扣
精度或性能阶段。
