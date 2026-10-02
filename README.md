# AgileX 双臂 pi 真机推理客户端

本仓库维护松灵双臂真机的同步远程 pi 推理执行代码。模型权重与推理服务部署在远端；本机采集三路相机和双臂状态，通过 WebSocket 请求动作 Chunk，再逐步提交给现有真机执行层。

当前只保留 pi 图像处理路径和同步推理。整理以初始提交 `854dc3f` 为行为基线，没有修改 ROS/CAN 控制算法，也没有修复下文列出的已知问题。

## 安装

使用 Python 3.10，推荐在仓库内建立虚拟环境并使用可编辑安装：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./packages/openpi-client
python -m pip install -e .
```

也可以使用已有的 uv：

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e ./packages/openpi-client
uv pip install -e .
```

先安装仓库内的 `openpi-client`，再安装主项目，确保使用本仓库验证过的同步 Broker。不要用上层 openpi 工程的另一个客户端替代它。

主项目依赖按整理前虚拟环境中实际使用的版本声明。ROS 和硬件 SDK 需要沿用真机已经验证过的安装环境：

- ROS 1 的 `rospy`、`sensor_msgs`、`geometry_msgs`、`nav_msgs`、`std_msgs` 和 `cv_bridge`。
- 与当前控制代码兼容的 Piper SDK；优先使用 `C_PiperInterface_V2`，代码保留旧接口回退。
- 在运行 shell 中加载 ROS 及机器人工作空间的 setup，使上述模块能被当前 Python 解释器导入。ROS 的 Python ABI 必须与解释器兼容，仅加载 setup 不能消除版本不匹配。
- 已启动的相机节点；位置控制还需要机器人 ROS 状态和指令节点，MIT 控制需要已经配置的 CAN 接口。

不要为此次整理升级真机 SDK、固件或 ROS 环境。安装和查看 CLI 帮助本身不连接服务器、不初始化 ROS、不使能机械臂。

`requirements.in`、`requirements.txt` 和 `uv.lock` 保留为历史 ALOHA 环境参考，不是当前安装来源。请使用上述 pip/uv pip 命令，不使用该历史锁文件执行 `uv sync` 或 `uv run`。

## 启动

安装后可以在任意工作目录使用模块或命令入口：

```bash
python -m mobile_aloha_agilex.main --help
agilex-infer --help
```

在仓库根目录也可以运行 `python main.py --help`。三种入口使用相同的参数和实现。

位置控制：

```bash
agilex-infer --host SERVER_IP --port 8000 --control-mode position --action-horizon 30
```

MIT 控制：

```bash
agilex-infer --host SERVER_IP --port 8000 --control-mode mit \
  --can-port-left can_left --can-port-right can_right \
  --mit-kp 10.0 --mit-kd 0.8 --action-horizon 30
```

执行启动命令会连接模型服务器，随后初始化机器人，并在 episode 开始时执行现有复位流程。正常完成所有 episode 后，Runtime 再复位一次；异常退出沿用原有环境关闭流程。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--host` / `--port` | `10.19.134.58` / `8000` | 远端 pi WebSocket 服务 |
| `--action-horizon` | `30` | 每个 Chunk 实际执行的步数 |
| `--control-mode` | `position` | ROS 位置控制或 `mit` SDK/CAN 控制 |
| `--can-port-left` / `--can-port-right` | `can_left` / `can_right` | 双臂 CAN 接口 |
| `--mit-kp` / `--mit-kd` | `10.0` / `0.8` | MIT 增益 |
| `--num-episodes` | `1` | episode 数量 |
| `--max-episode-steps` | `10000` | 原有 Runtime 的步数限制参数，见已知问题 |

执行循环频率上限保留为 30 Hz。同步模式执行完指定步数后阻塞请求下一段动作，Chunk 之间存在网络和模型推理等待，实际频率可能更低。

旧的 `--async-infer`、`--trigger-step`、`--blend-window` 已移除，旧异步启动命令需要删除这些参数。原 `examples.mobile_aloha_AgileX` 导入路径改为 `mobile_aloha_agilex`。

推理入口的记录功能也暂时移除：不再支持 `--record`、`--record-rate`、`--record-file`。旧命令需要删除这些参数，启动和退出不会自动保存关节记录或绘图。

## 模型接口

所有 pi 系列服务使用同一套客户端协议；客户端不根据 metadata 的模型名称选择预处理，也不检查模型名称。服务端需要与以下现有接口兼容。

建立连接后，服务器首先发送 MessagePack 编码的 metadata。客户端保留 metadata 日志及 `reset_pose` 传递；实际真机复位仍由执行层的硬编码左右臂姿态决定。

每次推理请求：

- `state`：14 维双臂位置，顺序为左臂六关节、左夹爪、右臂六关节、右夹爪。
- `images`：`cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路图像。
- 图像沿用 `resize_with_pad → uint8 → CHW`，默认尺寸 `3×224×224`；深度字段被移除。
- 不额外发送 prompt、速度观测或 effort，也不改变 ROS 原图的颜色通道顺序。

服务器返回 `actions: (H, 14)`，可选 `velocities: (H, 14)`。H 必须不少于 `action_horizon`。同步 Broker 执行前 `action_horizon` 步后请求新 Chunk；返回的 Chunk 更长时，其余步骤不执行。

位置控制只传位置；MIT 模式保留可选速度透传，没有速度时沿用执行层默认处理。MIT 夹爪按米处理，ROS 路径仍直接透传原值，本次未统一单位。

## 独立监控工具

推理入口不再传递记录配置，底层记录实现保持关闭。`monitor/` 保留通用关节监控 `monitor_joints.py`、相机监控 `monitor_camera.py` 和已有数据；两个专用左臂记录脚本已移除。目前没有自动记录双臂的功能。

需要独立视频录制时另开终端：

```bash
python -m mobile_aloha_agilex.monitor.monitor_camera record
```

默认视频写入 `monitor/videos/`。监控工具也保留脚本入口，具体参数查看各脚本帮助。

`convert_aloha_data_to_lerobot.py` 是历史 LeRobot v2.0 数据转换辅助脚本，需要单独配置匹配其 API 的 LeRobot、Torch、h5py 等环境；推理安装不包含训练框架。

## 代码职责与维护

| 内容 | 职责 |
|---|---|
| `main.py` | CLI、同步推理组件组装与运行 |
| `env.py` | pi 观测预处理及模型动作到执行层的适配 |
| `real_env.py`、`robot_utils.py`、`mit_controller.py`、`constants.py` | 原有真机执行与 ROS/CAN 交互 |
| `packages/openpi-client/` | 同步分块、传输、序列化和 Runtime |
| `monitor/` | 关节与相机监控工具；按需导入，不在包导入时启动 |
| `tests/` | 离线行为对照、接口和入口验证 |

硬件层仍保留部分未参与推理的旧函数和 RDT 配置，控制逻辑保持不变；必要导入已调整，已删除的左臂记录工具明确禁用。`new/` 重复实现、异步 Broker、旧 ALOHA Docker/compose 和未接入的显示器已移除，历史内容可以从提交 `854dc3f` 恢复。

虚拟环境、实验录制数据、视频、模型权重和缓存由 Git 忽略，整理没有删除已有实验数据。目录是独立 Git 仓库，维护时在这里提交；上层工程不会自动记录本仓库内部修改。

## 离线验证

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

默认只收集 `tests/` 和客户端测试。测试使用固定图像、假策略和替身环境，不连接实际机器人。

行为对照测试以 Git 中的 `854dc3f` 为基线；没有该提交历史的源码副本会跳过相关对照测试。真机控制文件去除导入后的 AST 必须与基线一致，同步 Broker、WebSocket 和 Runtime 实现必须保持一致。

如果本机已安装 uv 且所需依赖已缓存，可以额外运行独立安装检查：

```bash
python tests/check_standalone.py
```

该检查将源码复制到临时目录，离线建立全新虚拟环境，验证无上层工程和硬件依赖时的包导入及三种帮助入口；结束后自动移除临时环境，不连接机器人。

## 后续待修复问题

本次等价整理保留以下行为：

1. 同步 Broker 未校验 Chunk 长度、动作维度或 NaN/Inf；短 Chunk 可能在部分动作执行后越界。
2. metadata 和推理响应接收没有超时；只有连接被拒绝时持续重试，其他连接异常直接传播。
3. WebSocket 没有显式关闭接口，初始化失败时的资源清理仍不完整。
4. 观测预处理原地删除深度字段并替换图像；重复处理同一份观测可能出错。
5. Runtime 的步数限制可能比指定值多执行一步。
6. 服务器 `reset_pose` 未用于实际复位；ROS 与 MIT 夹爪单位没有统一。
7. MIT 指令失败返回值、使能状态和读状态失败的处理沿用现有代码。

这些项目应作为后续单独修复和真机验收任务，不能用本次离线测试结果替代实际设备验证。
