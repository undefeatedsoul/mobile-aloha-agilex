"""Run synchronous pi inference on an AgileX dual-arm robot."""

import dataclasses
import logging

import tyro


@dataclasses.dataclass
class Args:
    # Remote pi policy server.
    host: str = "10.19.134.58"
    port: int = 8000
    # Number of actions executed before requesting another chunk.
    action_horizon: int = 30
    # Number of episodes to run.
    num_episodes: int = 1
    # Episode step limit using the existing Runtime counting behavior.
    max_episode_steps: int = 10000

    # Control mode: position (ROS) or mit (Piper SDK/CAN).
    control_mode: str = "position"
    # Left-arm CAN interface used in MIT mode.
    can_port_left: str = "can_left"
    # Right-arm CAN interface used in MIT mode.
    can_port_right: str = "can_right"
    # MIT position gain.
    mit_kp: float = 10.0
    # MIT velocity gain.
    mit_kd: float = 0.8


def main(args: Args) -> None:
    # Import the inference stack only after parsing CLI arguments.
    from openpi_client import action_chunk_broker
    from openpi_client import websocket_client_policy
    from openpi_client.runtime import runtime
    from openpi_client.runtime.agents import policy_agent

    from mobile_aloha_agilex import env

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    mit_config = None
    if args.control_mode == "mit":
        mit_config = {
            "can_port_left": args.can_port_left,
            "can_port_right": args.can_port_right,
            "kp": args.mit_kp,
            "kd": args.mit_kd,
        }
        logging.info("MIT control config: %s", mit_config)

    broker = action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=args.action_horizon, chunk_sleep=0)
    environment = env.AlohaRealEnvironment(
        reset_position=metadata.get("reset_pose"),
        control_mode=args.control_mode,
        mit_config=mit_config,
    )
    runner = runtime.Runtime(
        environment=environment,
        agent=policy_agent.PolicyAgent(policy=broker),
        subscribers=[],
        max_hz=30,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )
    try:
        runner.run()
    finally:
        try:
            environment.close()
        except Exception as exc:
            logging.error("关闭环境失败: %s", exc)


def cli() -> None:
    """Shared entry point for the module, script and installed command."""
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
