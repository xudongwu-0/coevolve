"""Explicit role credit, separated from the environment's unchanged outcome."""


def role_reward(role: str, team_reward: float, model_error, mode: str = "team", *,
                webshop_receipt=None) -> float:
    if role not in ("editor", "solver") or (
        webshop_receipt is None and team_reward not in (0.0, 1.0)
    ):
        raise ValueError("Invalid joint role or binary environment outcome")
    if mode not in ("team", "validity_gated"):
        raise ValueError("Unknown Engineer credit mode")
    if mode == "validity_gated" and role == "editor" and model_error is not None:
        if webshop_receipt is not None:
            from .webshop_reward import verified_reward
            if team_reward != verified_reward(webshop_receipt):
                raise ValueError("Joint reward differs from the verified environment receipt")
        return 0.0
    if webshop_receipt is not None:
        from .webshop_reward import verified_reward
        if team_reward != verified_reward(webshop_receipt):
            raise ValueError("Joint reward differs from the verified environment receipt")
    return float(team_reward)
