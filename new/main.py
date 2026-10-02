import dataclasses
import logging
import os
import subprocess
import signal
import time
import atexit
import json
from typing import Optional

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.mobile_aloha_AgileX import env as _env

# 同步文件路径
SYNC_FILE = "/tmp/monitor_sync_ready"
MONITOR_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "monitor", "monitor_joints_left_mit.py"
)
MONITOR_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "monitor", "datas"
)

# 全局记录配置 (供 real_env.py 读取)
RECORD_CONFIG = {
    "enabled": False,
    "start_time": None,
    "output_file": None,
}


@dataclasses.dataclass
class Args:
    host: str = "10.22.19.4"
    #host: str = "10.19.134.58"
    port: int = 8000
    action_horizon: int = 50 #25

    num_episodes: int = 1
    max_episode_steps: int = 10000
    
    # 控制模式配置
    control_mode: str = "position"  # "position" 或 "mit"
    
    # MIT 控制参数 (SDK建议值: kp=10, kd=0.8)
    can_port_left: str = "can_left"    # 左臂CAN端口
    can_port_right: str = "can_right"  # 右臂CAN端口
    mit_kp: float = 10.0  # 位置增益，SDK建议值10，范围[0, 500]
    mit_kd: float = 0.8   # 速度增益，SDK建议值0.8，范围[-5, 5]
    
    # 关节数据记录 (仅MIT模式有效)
    record: bool = False  # 是否同步启动 monitor 记录关节数据
    record_rate: float = 50.0  # monitor 采样率 (Hz)
    record_file: str = ""  # 保存文件名 (不含路径和后缀)，为空则自动用时间戳命名


def main(args: Args) -> None:
    monitor_proc = None
    
    # 如果开启记录且是MIT模式，启动 monitor 子进程
    if args.record and args.control_mode == "mit":
        monitor_proc = start_monitor_process(args)
        if monitor_proc is None:
            logging.warning("Monitor 进程启动失败，继续运行但不记录数据")
    
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    
    # 从服务器 metadata 获取模型类型，用于区分图像处理分支
    model_type = metadata.get("model_type", "pi")  # 默认为 pi 保持后向兼容
    logging.info(f"Detected model type from server: {model_type}")
    
    # 构建MIT配置
    mit_config = None
    if args.control_mode == "mit":
        mit_config = {
            "can_port_left": args.can_port_left,
            "can_port_right": args.can_port_right,
            "kp": args.mit_kp,
            "kd": args.mit_kd,
        }
        logging.info(f"MIT control mode enabled with config: {mit_config}")
    
    # 构建记录配置 (传递给环境)
    record_config = None
    if args.record and args.control_mode == "mit":
        record_config = RECORD_CONFIG.copy()  # 传递当前配置的副本
    
    runtime = _runtime.Runtime(
        environment=_env.AlohaRealEnvironment(
            reset_position=metadata.get("reset_pose"),
            model_type=model_type,
            control_mode=args.control_mode,
            mit_config=mit_config,
            record_config=record_config,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
                chunk_sleep=0,  # 执行完一个 chunk 后等待 0 秒
            )
        ),
        subscribers=[],
        max_hz=30,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        # 关闭环境（这会触发 MasterRecorder.save() 保存 master 数据）
        try:
            if hasattr(runtime, '_environment'):
                runtime._environment.close()
        except Exception as e:
            logging.error(f"关闭环境时出错: {e}")
        
        # 停止 monitor 进程
        if monitor_proc is not None:
            stop_monitor_process(monitor_proc)


def start_monitor_process(args: Args):
    """
    启动 monitor 子进程，并等待同步
    
    同步流程:
    1. main 创建 SYNC_FILE 文件，写入 JSON (含 start_time, output_file)
    2. 启动 monitor 子进程 (sync 模式)
    3. monitor 读取 SYNC_FILE 获取共享时间戳，然后删除该文件
    4. main 检测到 SYNC_FILE 被删除后继续执行
    """
    from datetime import datetime
    global RECORD_CONFIG
    
    # 生成输出文件名
    if args.record_file:
        output_file = os.path.join(MONITOR_OUTPUT_DIR, f"{args.record_file}.csv")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(MONITOR_OUTPUT_DIR, f"inference_left_{timestamp}.csv")
    
    # 记录共享的启动时间戳
    start_time = time.time()
    
    # 创建同步文件 (JSON格式)
    os.makedirs(os.path.dirname(SYNC_FILE), exist_ok=True)
    os.makedirs(MONITOR_OUTPUT_DIR, exist_ok=True)
    sync_data = {
        "start_time": start_time,
        "output_file": output_file,
    }
    with open(SYNC_FILE, 'w') as f:
        json.dump(sync_data, f)
    
    # 设置全局配置 (供 real_env.py 使用)
    RECORD_CONFIG["enabled"] = True
    RECORD_CONFIG["start_time"] = start_time
    RECORD_CONFIG["output_file"] = output_file
    
    # 启动 monitor 子进程
    cmd = [
        "python", MONITOR_SCRIPT,
        "sync",  # 同步模式
        "-f", output_file,
        "-c", args.can_port_left,
        "-r", str(args.record_rate),
        "--sync-file", SYNC_FILE,
    ]
    
    logging.info(f"启动 monitor 进程: {' '.join(cmd)}")
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # 注册退出时清理
        atexit.register(lambda: stop_monitor_process(proc))
        
        # 等待同步 (monitor 删除 SYNC_FILE 表示已准备好)
        logging.info("等待 monitor 准备就绪...")
        timeout = 10.0  # 最多等待 10 秒
        start = time.time()
        while os.path.exists(SYNC_FILE):
            if time.time() - start > timeout:
                logging.error("Monitor 同步超时")
                proc.terminate()
                return None
            time.sleep(0.1)
        
        logging.info(f"Monitor 已就绪，数据将保存到: {output_file}")
        return proc
        
    except Exception as e:
        logging.error(f"启动 monitor 失败: {e}")
        if os.path.exists(SYNC_FILE):
            os.remove(SYNC_FILE)
        return None


def stop_monitor_process(proc):
    """停止 monitor 子进程"""
    if proc is None:
        return
    
    try:
        # 发送 SIGINT (相当于 Ctrl+C)，让 monitor 正常保存数据
        logging.info("停止 monitor 进程...")
        proc.send_signal(signal.SIGINT)
        
        # 等待进程结束
        try:
            proc.wait(timeout=5.0)
            logging.info("Monitor 进程已正常退出")
        except subprocess.TimeoutExpired:
            logging.warning("Monitor 进程未响应，强制终止")
            proc.kill()
            proc.wait()
    except Exception as e:
        logging.error(f"停止 monitor 失败: {e}")


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)

# python -m examples.mobile_aloha_AgileX.main.py
