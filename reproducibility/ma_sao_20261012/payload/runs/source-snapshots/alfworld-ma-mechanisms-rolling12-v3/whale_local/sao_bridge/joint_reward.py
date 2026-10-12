"""Explicit role credit, separated from the environment's unchanged outcome."""


def role_reward(role: str, team_reward: float, model_error, mode: str = "team") -> float:
    if role not in ("editor", "solver") or team_reward not in (0.0, 1.0):
        raise ValueError("Invalid joint role or binary environment outcome")
    if mode not in ("team", "validity_gated"):
        raise ValueError("Unknown Engineer credit mode")
    if mode == "validity_gated" and role == "editor" and model_error is not None:
        return 0.0
    return float(team_reward)
