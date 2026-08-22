# EC616 kinematics (vendored)

Self-contained EC616 / EA66 forward and inverse kinematics for `act_robot`.

Vendored from the `demo_test` project (numerical IK via SciPy `least_squares`,
standard DH chain). **Do not import external `demo_test` at runtime** — use
`ec616_kin` or `scripts/ec616_ik.py` only.

Dependencies: `numpy`, `scipy`.
