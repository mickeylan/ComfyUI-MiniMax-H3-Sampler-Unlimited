# new-aigc-comfyui-minimax-h3 接入 HR Endless Sampler 详细方案

本文面向 `J:\mickeylan\ai\new-aigc-comfyui-minimax-h3` 的维护 Agent。本文只定义对接边界和验收要求；HR Endless 仓库不修改平台的 Go、Vue、SQLite 或 ComfyUI fork。

## 1. 目标与边界

目标是为平台增加**单个导演镜头超过 MiniMax H3 原生约 15 秒限制**的能力。一个平台 Shot 仍是一个导演镜头；HR 的 physical chunks 只是串行低显存采样窗口，不是新镜头。

平台负责：

- Shot/连续 beat 的内容、时长、起止状态、摄影机约束和人工审核；
- 可见、仅发声、仅提及和未来未登场角色分类；
- 角色、造型、场景、道具和共享素材的稳定资产 ID；
- 最终参考图选择、上传顺序、Prompt 审批、过期传播和任务调度；
- 原生短镜头和 Endless 长镜头的产品入口。

HR Endless 负责：

- H3 `17k+5` 帧对齐；
- physical chunk 规划、串行采样、Video1/Audio1 续接和低显存生命周期；
- 将审核计划裁剪为当前 chunk 的局部六字段 Prompt；
- 按当前区间过滤、重新编号 Picture 引用及对应 Qwen/H3 图片引用；
- 事件账本、Qwen/Gemma观察、replay、retake、continuation、preview和timeline。

平台不得把 physical chunk 当作 Shot 保存或显示，也不得自行计算 H3 latent token、packing prefix 或音频 latent 边界。

## 2. HR 公开接入协议

### 2.1 ComfyUI custom type

```text
HR_H3_PROMPT_PLAN
```

当前 schema version：`1`。

`HR H3 Prompt Skill Compiler` 会输出该类型；平台也可以通过适配节点直接构造同一 payload，不必再次运行 Compiler。

### 2.2 v1 payload

```json
{
  "type": "HR_H3_PROMPT_PLAN",
  "version": 1,
  "fps": 24.0,
  "total_frames": 634,
  "image_subjects": [
    {
      "picture": 1,
      "subject": 1,
      "name": "角色名",
      "observable_features": "只写可见身份特征"
    }
  ],
  "shots": [
    {
      "start_frame": 0,
      "end_frame": 145,
      "pictures": [1, 2, 4],
      "cut": true,
      "camera": "rear three-quarter tracking, fixed axis",
      "start_state": "可见起始状态",
      "events": [
        {"id": "shot-101.B1.V1", "action": "一个原子可见动作", "phase": "start"}
      ],
      "end_state": "可供下一段继承的可见结束状态",
      "forbidden_replays": ["不得重演的既有动作或构图"],
      "audio": "当前区间实际声音",
      "description": "当前区间可执行的 H3 视觉描述"
    },
    {
      "start_frame": 145,
      "end_frame": 272,
      "pictures": [1, 2, 4, 5],
      "cut": false,
      "camera": "continue the same tracking take and axis",
      "start_state": "继承上一段结束状态",
      "events": [
        {"id": "shot-101.B2.V1", "action": "继续接近入口", "phase": "continue"}
      ],
      "end_state": "抵达入口",
      "forbidden_replays": ["不得重新开始奔跑"],
      "audio": "连续脚步和环境声",
      "description": "同一长镜头内的第二个 beat"
    }
  ],
  "summary": "仅用于审核的全局摘要",
  "retention_analysis": "仅用于审核的全局约束",
  "overall_soundscape": "仅用于审核的全局声音计划",
  "non_diegetic_music": "N/A",
  "warnings": []
}
```

### 2.3 帧范围语义

所有计划范围一律是半开区间：

```text
[start_frame, end_frame)
```

要求：

- 第一段从 `0` 开始；
- 相邻段严格连续；
- 最后一段 `end_frame == total_frames`；
- `fps` 和 `total_frames` 必须与 Sampler 输入及 H3 latent 完全一致；
- `total_frames` 必须位于 H3 的 `17k+5` 网格；
- 平台若读取 HR timeline 的旧 inclusive `end` 字段，转换公式是 `end_frame = end + 1`。

不满足时 HR 会在采样前拒绝执行，不做猜测或自动修复。

### 2.4 `cut` 语义

- `cut: true`：这是实际导演切镜，允许/要求 H3 Shot marker；
- `cut: false`：这是同一长镜头内部 beat，Qwen收到 `Continuous beat (no cut)`，不得生成新 Shot marker、切机位或重置空间关系；
- 缺失时默认 `true`，保持当前 Compiler 和旧工作流语义。

平台的一个长 Shot 可以被拆成多个 `cut:false` beat。physical chunk 边界不需要出现在该 payload 中。

## 3. Picture 和资产映射

平台数据库必须继续保存稳定资产 ID，例如：

```text
character:12:sheet
outfit:31:image
asset:8:image
shared_material:5:ref
```

不要把 `<Picture N>` 作为数据库长期身份。任务提交时：

1. 根据本次最终上传顺序创建稳定资产 ID → 全局 Picture 序号映射；
2. `image_subjects.picture` 和每段 `pictures` 使用该全局序号；
3. `Reference Set` 的图片连接顺序必须与全局序号一致；
4. HR 每个 physical chunk 对当前区间的 `pictures` 求并集；
5. HR 过滤不活跃图片，并把活跃图片局部重新编号为连续的 `<Picture 1..N>`；
6. HR 同步重写局部 `subject_definitions`、`detailed_description`、Qwen图片展示和H3 `minimax_refs`；
7. 视频/音频引用不按 Picture 规则删除。

例如当前区间只允许全局Pictures `[1, 3, 5]`，运行时局部映射为：

```text
Global Picture 1 → Local Picture 1
Global Picture 3 → Local Picture 2
Global Picture 5 → Local Picture 3
```

未来未登场主体的名字、描述、声音和图片都不得进入当前chunk。这是防止后期角色提前出现的硬约束，不应只依赖“不要提前出现”的自然语言。

## 4. 平台端建议数据扩展

不要求平台复制HR内部模型，但建议在现有 `Shot` 上增加或等价表达：

```text
generation_mode: native | endless | auto
continuous_take: bool
endless_chunk_frames: int（可选高级项）
director_policy: locked | conservative | adaptive | every_chunk
```

长镜头内部 beat 建议作为独立的 ShotBeat 表或版本化 JSON 保存，至少包含：

```text
stable beat id
start/end seconds or duration
cut
camera contract
start/end state
events
forbidden replays
visible asset ids
audio
description
```

平台UI以秒显示和编辑，提交适配器负责转换到帧并处理总帧H3对齐。最后一个beat吸收对齐产生的尾部帧，不能在中间累计舍入误差。

## 5. 模式选择

建议平台保留两条路径：

### native

- 3–15秒普通镜头；
- 使用现有MiniMax H3模板；
- 保留多GPU并行优势。

### endless

- 单个导演镜头超过15秒；
- 用户手动指定；
- 需要chunk断点、内部重拍或低显存；
- 同一Shot内部串行，但不同Shot仍可由平台并行调度。

`auto`建议只做推荐，不应静默改变用户已审核的执行模式：

```text
Duration > 15s → 提示推荐Endless，由用户确认
```

## 6. ComfyUI工作流接线

平台新增一个Endless模板，核心连接：

```text
平台计划适配节点.prompt_plan
  → HR Endless Sampler.prompt_plan

平台计划适配节点.initial_event_ledger
  → HR Endless Sampler.initial_event_ledger

平台编译的完整审核Prompt
  → HR Endless Sampler.prompt

Reference Set
  → Reference Conditioning
  → BasicGuider / latent
  → HR Endless Sampler.reference_set

HR Endless Sampler.timeline
  → HR Endless Sampler Save Video
```

`prompt`仍必须连接：

- 作为人工审核展示和旧路径兼容；
- 提供完整原始意图及marker解析兜底；
- `prompt_plan`连接时，实际每块编码前会被严格局部化。

## 7. Prompt Skill Compiler和平台适配器

### 独立ComfyUI模式

```text
普通故事
→ HR H3 Prompt Skill Compiler
→ H3 prompt + HR_H3_PROMPT_PLAN + initial event ledger
→ HR Endless Sampler
```

### 平台模式

```text
SQLite中已审核Scene/Shot/Beat/资产
→ Go端构造versioned plan
→ 工作流适配节点输出HR_H3_PROMPT_PLAN
→ HR Endless Sampler
```

平台模式不应再次调用Prompt Skill Compiler覆盖人工审核数据。Compiler只作为独立用户入口或平台“生成草稿”的参考实现。

## 8. Director策略

平台可传递策略，第一版至少实现：

- `locked`：完全依照审核计划；不允许Qwen增加动作、角色、运镜或切镜；
- `conservative`：推荐默认，只在beat边界/状态不确定时观察；
- `adaptive`：允许根据已生成画面调整beat进度，但不能改变最终状态；
- `every_chunk`：保持当前每physical chunk调用Qwen，最慢，仅供困难镜头和诊断。

当前HR仍支持现有Qwen/Gemma每块Director。平台上线前应减少不必要调用：26秒、39帧chunk约19块，如果每块Qwen3.8耗时2–4分钟，生产成本不可接受。

## 9. 与 minimaxh3-director 的兼容

HR必须继续支持现有链路：

```text
MiniMaxH3DirectorCS.model → Preview/Guider/Scheduler
MiniMaxH3DirectorCS.positive → BasicGuider
MiniMaxH3DirectorCS.latent → HR Endless Sampler.latent_image
MiniMaxH3DirectorCS.prompt → HR Endless Sampler.prompt
MiniMaxH3DirectorCS.fps → HR Endless Sampler.fps
```

兼容规则：

- `prompt_plan`是可选输入；不连接时走原有纯Prompt解析和DirectorCS conditioning，行为不变；
- 新输出追加在Prompt Skill Compiler输出末尾，不改变原有输出索引；
- 不修改`MiniMaxH3DirectorCS`节点、serialized properties或timeline_data；
- 可继续制作“DirectorCS负责时间线，HR负责physical chunks”的测试工作流；
- 同一测试中若同时连接DirectorCS和`prompt_plan`，typed plan仅局部化文本/图片引用，DirectorCS仍提供模型、latent和基础conditioning；两边帧数和Picture顺序必须一致，否则执行前拒绝。

## 10. Replay、重拍和过期规则

计划内容必须参与replay身份：

- Prompt或plan变化时，旧的Qwen timing plan不可直接复用；
- 可以保留已完成physical chunk作为视觉边界，但后续chunk必须基于新plan重新导演；
- 平台应保存plan schema version、plan hash和资产映射hash；
- Shot/Beat时长、事件、参考资产、Prompt或连续性输入变化时，平台必须标记旧视频候选过期；
- 不要让平台依赖HR临时目录或`.pt`内部结构，只保存任务ID、输出文件、timeline sidecar和HR公开checkpoint ID。

## 11. Timeline回传

平台应保存：

```text
任务ID
HR timeline sidecar
plan schema version/hash
每个chunk的实际runtime prompt
chunk帧范围和耗时
涉及的Shot/Beat stable IDs
active asset IDs/Picture映射
Qwen调整报告
replay/checkpoint引用
```

平台UI中默认显示平台Shot和beat；physical chunk只放在高级诊断/重拍界面，不应伪装成导演镜头。

## 12. 分阶段对接

### Phase A：模板和协议

- 安装HR插件；
- 新增Endless工作流模板；
- Go适配器生成`HR_H3_PROMPT_PLAN v1`；
- 打通一个20–30秒单Shot；
- 保存timeline和plan hash。

### Phase B：资产门控

- 对每个beat输出稳定资产ID；
- 提交时映射全局Picture顺序；
- 验证未来角色的文本、Qwen图片和H3 refs都不进入早期chunk；
- 验证跨beat chunk使用两边资产并集且局部编号正确。

### Phase C：审核与策略

- Shot编辑器加入Endless开关和beat折叠编辑；
- 显示计划总时长/H3对齐帧数/预计chunk数；
- 增加locked/conservative/adaptive/every_chunk；
- Prompt预览分别显示审核全局Prompt和实际chunk Prompt。

### Phase D：恢复与重拍

- 接入checkpoint/replay；
- 平台按chunk发起内部重拍；
- 验证平台/ComfyUI重启恢复；
- 将HR timeline映射到平台候选和版本系统。

## 13. 必须通过的验收矩阵

### 时序与泄漏

- 26.042秒、24fps对齐为634帧；
- `chunk_frames=39`应产生19个physical chunks；
- 后期角色只属于后期beat时，Chunk 1最终Prompt、Qwen图片和H3 refs中都不存在该角色；
- 后期角色首次登场chunk才加入其局部Picture；
- `cut:false` beat边界不生成新Shot marker或改变机位轴线；
- `cut:true`边界只生成一次真实切镜。

### 连续性

- 运行、滑行、拥抱、对白等跨chunk动作不重启；
-上一chunk结束状态成为下一chunk起始状态；
-已完成事件进入forbidden，不再重演；
-已登场但当前离屏角色可保留状态，但不得被提示为当前可见。

### 媒体

- Picture局部重编号与实际Qwen/H3图片顺序完全一致；
- 9张图片上限、3个视频及同索引音轨、3条独立音频保持有效；
- Video1/Audio1续接不因Picture筛选错位；
-音视频总长度和timeline总帧一致。

###兼容

-旧纯Prompt工作流不连接`prompt_plan`时结果路径不变；
- `MiniMaxH3DirectorCS`旧工作流能加载和采样；
- Prompt Skill Compiler既有输出索引不变；
- replay、retake、continuation和Save/Load timeline继续工作。

###性能

-记录每chunk H3、Qwen、VAE和总耗时；
-比较every_chunk与conservative调用次数；
-验证12GB路径峰值显存；
-验证8×L40平台中不同Shot可并行，而单个Endless Shot内部保持串行。

## 14. 已知边界

- HR typed plan是执行协议，不是平台数据库schema；平台应有自己的持久模型和适配器；
- `HRENDLESS_TIMELINE`现有范围是inclusive，平台业务计划应保持half-open并在边界转换；
-真实GPU/完整ComfyUI验收仍必须在目标部署环境完成；
-平台不可依赖HR私有函数、replay文件名、临时目录、tensor字典或Qwen worker内部payload。

## 15. 对接完成定义

只有同时满足以下条件才算接入完成：

1. 平台一个大于15秒的单Shot可以选择Endless并生成一个连续视频；
2. 平台Shot和beat仍是唯一导演事实来源；
3. physical chunks不变成平台Shot；
4. 未来角色、声音、事件和参考图不会提前进入；
5. `cut:false`保持单镜头，`cut:true`只在真实导演边界切镜；
6. 用户能审核全局计划并回看实际chunk Prompt；
7. 旧原生短镜头、多GPU调度和DirectorCS测试工作流不受影响；
8. 断点、重拍、timeline和最终媒体元数据可由平台追踪。
