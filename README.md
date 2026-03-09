cat > /home/siva/orchid_project/orchid/README.md << 'EOF'
# ORCHID
## Orchestrated Resilient Coordination for Heterogeneous In-orbit Debris avoidance

A multi-agent reinforcement learning framework for autonomous satellite collision avoidance with unknown partner behaviours.

## Key Result
Zero collisions against unseen partner policies (zero-shot generalisation):
| Partner Policy | Seen During Training | Reward | Collisions |
|---|---|---|---|
| ProgradePolicy | ❌ No | 67.9 | 0 |
| RetrogradePolicy | ❌ No | 65.2 | 0 |

## Installation
```bash
git clone https://github.com/Sivaram-Venugopal/ORCHID.git
cd ORCHID
python3 -m venv orchid_env
source orchid_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3
```

## Train
```bash
PYTHONPATH=. python3 training/rpt_train.py
```

## Evaluate
```bash
PYTHONPATH=. python3 evaluation/zero_shot_eval.py
```

## Project Structure
- `env/` — Orbital physics simulation (J2 perturbations, RK4 propagation)
- `partners/` — Partner policy pool (7 training + 2 held-out eval policies)
- `agents/` — Safety layer with collision probability filtering
- `training/` — PPO training with Robust Policy Training
- `evaluation/` — Zero-shot evaluation framework

## Author
Sivaram Venugopal
EOF
