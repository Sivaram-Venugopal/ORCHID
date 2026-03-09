import sys
sys.path.insert(0, '/home/siva/orchid_project/orchid')

import numpy as np
from stable_baselines3 import PPO
from env.orbital_env import OrchestratedOrbitEnv, EnvConfig
from partners.policy_pool import get_all_eval_policies

INIT_FUEL = 50.0

def evaluate_policy(model, partner_policy, n_episodes=20):
    config = EnvConfig()
    env = OrchestratedOrbitEnv(config=config, partner_policy=partner_policy)
    
    rewards = []
    fuels_used = []
    collisions = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_collisions = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_collisions += 1 if info.get('collision', False) else 0
        
        fuel_used = INIT_FUEL - float(info.get('fuel', INIT_FUEL))
        rewards.append(ep_reward)
        fuels_used.append(fuel_used)
        collisions.append(ep_collisions)
    
    return {
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'fuel_mean': np.mean(fuels_used),
        'collisions_total': np.sum(collisions)
    }

def main():
    print("Loading trained model...")
    model = PPO.load("models/orchid_ppo.zip")
    print("Model loaded!\n")
    
    eval_policies = get_all_eval_policies()
    
    print("=" * 65)
    print("ORCHID ZERO-SHOT EVALUATION RESULTS")
    print("=" * 65)
    print(f"{'Policy':<22} {'Reward':>10} {'Std':>8} {'Fuel(kg)':>10} {'Collisions':>12}")
    print("-" * 65)
    
    for policy in eval_policies:
        name = policy.__class__.__name__
        results = evaluate_policy(model, policy, n_episodes=20)
        seen = "❌" if name in ['ProgradePolicy', 'RetrogradePolicy'] else "✅"
        print(f"{seen} {name:<20} {results['reward_mean']:>10.1f} {results['reward_std']:>8.1f} {results['fuel_mean']:>10.2f} {results['collisions_total']:>12}")
    
    print("=" * 65)
    print("\n✅ = seen during training   ❌ = never seen (zero-shot)")
    print("\nDone!")

if __name__ == "__main__":
    main()
