class MAPPOPolicyStub:
    def __init__(self, shared_policy: bool = True) -> None:
        self.shared_policy = shared_policy

    def act(self, actor_obs, action_mask=None):
        if action_mask is None:
            return {"action": 0, "log_prob": 0.0, "value": 0.0}

        valid_actions = [idx for idx, flag in enumerate(action_mask) if bool(flag)]
        action = valid_actions[0] if valid_actions else 0
        return {"action": action, "log_prob": 0.0, "value": 0.0}
