# CLAUDE.md

This file is automatically read by Claude and other AI coding assistants.

<!-- Import the shared working rules -->
See [AGENTS.md](./AGENTS.md) for the full working rules that apply to this repository.

## Quick Reference for AI Agents

- **Frozen schema**: `schema-v1.0-frozen`, 19-channel fixed-mast, `meta/info.json` is single source of truth.
- **Camera keys**: `camera_wrist_left`, `camera_wrist_right`, `head`, `base`.
- **Do NOT** write safety/watchdog/estop code — see AGENTS.md.
- **Do NOT** hardcode channel counts; always load from `meta/info.json`.
- **Every test skip** must cite a GitHub issue: `# openAMRobot/openamrobot-c-baseline#16`.
- **Partner build** is always written as `Use_Case_1`.
- **Python target**: 3.10/3.12 compatible syntax only (ROS 2 Jazzy).
