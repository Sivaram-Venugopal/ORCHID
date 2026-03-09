import numpy as np
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.orbital_env import OrchestratedOrbitEnv, EnvConfig
from partners.policy_pool import get_random_training_policy, get_all_eval_policies

# Limit CPU threads for 4GB RAM
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"

LOCAL_CONFIG = {
    "n_envs":           1,
    "n_steps":          512,
    "batch_size":       32,
    "total_timesteps":  10_000,
}

def make_env(seed=0):
    cfg = EnvConfig(n_controlled=1, n_partner=1, n_debris=5, episode_len_s=3600)
    partner = get_random_training_policy()
    return OrchestratedOrbitEnv(cfg, partner_policy=partner, seed=seed)

def train(total_timesteps=None, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    steps = total_timesteps or LOCAL_CONFIG["total_timesteps"]

    env = make_vec_env(make_env, n_envs=LOCAL_CONFIG["n_envs"])

    model = PPO(
        "MlpPolicy", env,
        n_steps=LOCAL_CONFIG["n_steps"],
        batch_size=LOCAL_CONFIG["batch_size"],
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    print(f"Starting training for {steps} timesteps...")
    model.learn(total_timesteps=steps)
    model.save(f"{output_dir}/orchid_ppo")
    print(f"Model saved to {output_dir}/orchid_ppo.zip")
    env.close()
    return model

def evaluate(model_path="models/orchid_ppo", n_episodes=5):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    eval_policies = get_all_eval_policies()

    for policy in eval_policies:
        policy_name = policy.__class__.__name__
        cfg = EnvConfig(n_controlled=1, n_partner=1, n_debris=5, episode_len_s=3600)
        env = OrchestratedOrbitEnv(cfg, partner_policy=policy, seed=99)
        rewards = []
        collisions = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            total_collisions = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(action)
                total_reward += r
                if info.get("collision", False):
                    total_collisions += 1
            rewards.append(total_reward)
            collisions.append(total_collisions)

        print(f"{policy_name}: avg_reward={np.mean(rewards):.1f}, avg_collisions={np.mean(collisions):.2f}")

if __name__ == "__main__":
    train()
