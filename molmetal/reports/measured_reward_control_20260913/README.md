# Retained launcher failure, excluded from the reward comparison

Both treatments completed six chemistry jobs but had zero physical/search
energy measurements because the launcher used the environment's Python path
without entering the uv-managed PATH. `mk_prepare_receptor.py` was therefore
unavailable. All errors and candidates are preserved. These outputs are not
measured docking comparisons and do not show a reward effect.

The same verified runtime snapshot is used by the corrected uv-launched run
in `../measured_reward_control_20260913_v2/`. No original output was relabelled
or overwritten to hide the launcher failure.
