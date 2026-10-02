"""Adapt the existing robot environment to the pi observation/action protocol."""

import logging

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment
from typing_extensions import override


class AlohaRealEnvironment(environment.Environment):
    """Three-camera, 14-dimensional adapter for synchronous remote pi inference."""

    def __init__(
        self,
        reset_position: list[float] | None = None,
        render_height: int = 224,
        render_width: int = 224,
        control_mode: str = "position",
        mit_config: dict | None = None,
    ) -> None:
        # The adapter can be imported offline; hardware is loaded on construction.
        from mobile_aloha_agilex import real_env

        self._env = real_env.make_real_env(
            init_node=True,
            reset_position=reset_position,
            control_mode=control_mode,
            mit_config=mit_config,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._control_mode = control_mode
        self._ts = None
        logging.info("Control mode: %s", control_mode)
        if control_mode == "mit":
            logging.info("MIT config: %s", mit_config)

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for name in list(obs["images"]):
            if "_depth" in name:
                del obs["images"][name]

        for name, image in obs["images"].items():
            image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, self._render_height, self._render_width)
            )
            obs["images"][name] = einops.rearrange(image, "h w c -> c h w")

        return {"state": obs["qpos"], "images": obs["images"]}

    @override
    def apply_action(self, action: dict) -> None:
        """Forward positions, optional MIT velocities and recording metadata."""
        positions = action["actions"]
        velocities = action.get("velocities")
        chunk_id = int(action.get("_chunk_id", -1))
        in_overlap = bool(action.get("_in_blend", False))
        chunk_meta = {
            "chunk_step": int(action.get("_chunk_step", -1)),
            "chunk_actions_full": action.get("_chunk_actions_full"),
            "chunk_velocities_full": action.get("_chunk_velocities_full"),
            "chunk_skip_steps": int(action.get("_chunk_skip_steps", 0)),
            "chunk_blend_window": int(action.get("_chunk_blend_window", 0)),
        }
        if self._control_mode == "mit" and velocities is not None:
            self._ts = self._env.step(
                positions, velocities=velocities, chunk_id=chunk_id, in_overlap=in_overlap, chunk_meta=chunk_meta
            )
        else:
            self._ts = self._env.step(positions, chunk_id=chunk_id, in_overlap=in_overlap, chunk_meta=chunk_meta)

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()
