# 遮挡实现与区域依据

区域依据来自任务 0 的真实 BDDL goal：`(On akita_black_bowl_1 plate_1)`，因此目标是固定 placement destination `plate_1`，不是待抓的 bowl。

## 坐标与分割

用同一 MuJoCo 渲染 context 获取可见 geom ID。robosuite 1.4 在 NumPy 2 下解码 ID RGB 时发生 uint8×256 OverflowError；项目只在本地 helper 中将其转 int32 后解码，未修改上游文件。
原始 RGB 重渲染与 observation 的平均绝对像素差约 0.00126/255，其他翻转方向差异明显，确认视角一致；不要求因浮点渲染产生微小差异的两次 RGB 渲染逐像素位等。

先按 robosuite IMAGE_CONVENTION 对齐，再与官方 LiberoProcessorStep 一样旋转 180°。分割图用 nearest resize 至 512×512，对应 SmolVLA 的实际 resize。可见目标 mask 是所有 `plate_1_*` geom 的并集，仅取类型为 geom 的分割像素。

质心按可见像素中心计算；正方形边界是半开区间 `[x0,y0,x1,y1)`。从整数边长 1 起逐个搜索，返回第一块达到覆盖比例且完整位于画内的方形，不裁剪成矩形。

背景 anchor 在真实初始帧检查后设为 `[320,460]`，配置见 configs/regions_task0.json。每个 reset 都重新验证全部三档背景框只覆盖 table_visual 可见像素；不能覆盖 robot、目标、待抓物、其他物体/地标，也不能越界。anchor 在 Pilot 中冻结；更多状态可能不满足，必须报错并记录，不能静默移动 anchor。

## Policy hook

锁定 LeRobot v0.4.4，在单进程同步 `prepare_images` 调用范围内包装其 `resize_with_pad`。仅当 camera key 为 observation.images.image 时，在 resize 完成后、SigLIP `*2-1` 前替换为 RGB 128。保持原始 camera order，校验两个相机均实际调用，腕部图像原样返回。不更改权重、语言、状态、物理场景或录像来制造遮挡效果。

同一 episode 的 bbox 固定，每次实际推理都应用；模型 action queue 由官方 rollout 在每个 episode 开始 reset。processor pipelines 同样 reset，推理种子固定，条件顺序使用另一记录的种子随机化。

真实图像依据位于 artifacts/segmentation/task0-state0/；每次 rollout 会保存原始输入、mask 后真实 resize 输入、分割图、目标 overlay、三档成对 geometry 和真实动作/成功记录。actual RGB float.npy 保留 bilinear resize 后的实际浮点像素，PNG 仅用于查看。

单元测试覆盖：零面积不变、最小边长、T/B 同尺寸、像素面积、越界/非法背景拒绝、ID 解码，以及直接调用实际上游 prepare_images 验证遮挡位置、未遮挡区域和腕部输入不变。
