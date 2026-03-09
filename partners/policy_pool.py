import numpy as np

class BasePolicy:
    def act(self, obs, state):
        raise NotImplementedError

class PassivePolicy(BasePolicy):
    def act(self, obs, state):
        return np.zeros(3)

class AggressivePolicy(BasePolicy):
    MAX_DV = 0.015
    def act(self, obs, state):
        return np.array([self.MAX_DV, 0.0, 0.0])

class RuleBasedPolicy(BasePolicy):
    THRESHOLD_KM = 2.0
    DV = 0.005
    def act(self, obs, state):
        min_dist = obs[0] if len(obs) > 0 else 999.0
        if min_dist < self.THRESHOLD_KM:
            return np.array([0.0, self.DV, 0.0])
        return np.zeros(3)

class FuelMiserPolicy(BasePolicy):
    THRESHOLD_KM = 0.200
    DV = 0.003
    def act(self, obs, state):
        min_dist = obs[0] if len(obs) > 0 else 999.0
        if min_dist < self.THRESHOLD_KM:
            return np.array([0.0, self.DV, 0.0])
        return np.zeros(3)

class RandomPolicy(BasePolicy):
    MAX_DV = 0.015
    def __init__(self, burn_prob=0.1, seed=None):
        self.burn_prob = burn_prob
        self.rng = np.random.default_rng(seed)
    def act(self, obs, state):
        if self.rng.random() < self.burn_prob:
            return self.rng.uniform(-self.MAX_DV, self.MAX_DV, 3)
        return np.zeros(3)

class MirrorPolicy(BasePolicy):
    def __init__(self):
        self.last_dv = np.zeros(3)
    def act(self, obs, state):
        return self.last_dv.copy()
    def update_mirror(self, dv):
        self.last_dv = dv.copy()

class ProgradePolicy(BasePolicy):
    DV = 0.005
    def act(self, obs, state):
        return np.array([0.0, self.DV, 0.0])

class RetrogradePolicy(BasePolicy):
    DV = 0.005
    def act(self, obs, state):
        return np.array([0.0, -self.DV, 0.0])

TRAINING_POLICIES = [
    PassivePolicy,
    AggressivePolicy,
    RuleBasedPolicy,
    FuelMiserPolicy,
    lambda: RandomPolicy(burn_prob=0.1),
    lambda: RandomPolicy(burn_prob=0.3),
    MirrorPolicy,
]

EVAL_POLICIES = [
    ProgradePolicy,
    RetrogradePolicy,
]

def get_random_training_policy():
    idx = np.random.randint(len(TRAINING_POLICIES))
    return TRAINING_POLICIES[idx]()

def get_all_eval_policies():
    return [p() for p in EVAL_POLICIES]
