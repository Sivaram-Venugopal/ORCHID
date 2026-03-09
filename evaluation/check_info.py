import sys
sys.path.insert(0, '/home/siva/orchid_project/orchid')

from env.orbital_env import OrchestratedOrbitEnv, EnvConfig
from partners.policy_pool import ProgradePolicy

config = EnvConfig()
env = OrchestratedOrbitEnv(config=config, partner_policy=ProgradePolicy())
obs, _ = env.reset()

for i in range(5):
    action, _ = [env.action_space.sample()], None
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"Step {i+1} info keys: {info}")
    if terminated or truncated:
        break
