# AGENTS.md — OpenAMRobot 2.0 Workstream C Baseline

> **Imported from**: openAMRobot org standard (openamrobot-interfaces).  
> This file is required by Section 10 of the OpenAMRobot 2.0 Working Rules.  
> Every active repository carries `AGENTS.md` (the rules) and `CLAUDE.md` (which imports it).

## Working Rules (Section 10)

### Code Authorship
- **Do not** author safety, emergency stop, brake, or watchdog code with an AI agent.
- Route all base velocity commands through the arbitrated `/cmd_vel_nav` → velocity smoother → collision monitor pipeline (openamr-platform-sw #35). Never bypass safety locks.
- Route all arm commands through the standard `ros2_control` / MoveIt 2 action interfaces.

### Repository Governance
- Every PR must include: work package it closes (C10/C11), integration gate record, dependency declaration with license, exact provenance block (base SHA, head SHA, CLI commands, test counts), and a **Not verified** section.
- Every commit must carry `Signed-off-by:` (`git commit -s`) and the PR description must include an explicit AI disclosure statement.
- Any mention of the partner build is written strictly as `Use_Case_1`.

### Test Rules
- Every PR includes a test that **fails when the change is reverted** (reversion/mutation contract test).
- Skips in pytest must cite a tracking issue (e.g., `# openAMRobot/openamrobot-c-baseline#16`).
- Test suite must remain 100% green on `Ubuntu 24.04 + Python 3.10/3.12 + torch==2.5.1` (ROS 2 Jazzy target) — no Python 3.13-only syntax.

### Schema Freeze
- `openamrobot-fixed-mast-19dof` with tag `schema-v1.0-frozen` is the frozen baseline.
- Do NOT alter channel counts or camera keys without formal change-control sign-off (HC-12).
- `meta/info.json` is the **single source of truth** for all scripts. Scripts load channel names, dimensions, and camera keys dynamically from it — do not hardcode them across multiple files.

### C9 Camera Keys (Normative)
- `observation.images.camera_wrist_left` — wrist left
- `observation.images.camera_wrist_right` — wrist right
- `observation.images.head` — ZED-121210 scene camera (720×1280)
- `observation.images.base` — Orbbec Gemini 336L nav camera (480×640)

### Dependency Policy
- Declare every new or changed dependency with its license and source.
- Integration gate record: document what upstream packages were searched, what was chosen, and what was rejected.

### Partner Anonymity
- All references to the partner build use the name `Use_Case_1` — never the partner's actual name.

## Codeowners
- `@4u-79` — Embodied AI & Data, Division B (Workstream C: C-series deliverables)
- `@panthera-momagdii` — Software Lead / Division B Lead (reviewer for C11)
