We run the following experiments, the goal is construct a table of policy X algorithm

i.e we run each policy model @agents/policies (MLP - GNN - AC) against each algorithm @agents/trainers (REINFROCE - PPO - GRPO - SAC)


# Baselines

python train.py --agent random  --config small --seed 42 --no-wandb
python train.py --agent greedy  --config small --seed 42 --no-wandb

# MLP x all trainers

python train.py --policy mlp --trainer reinforce --config small --seed 42 --no-wandb
python train.py --policy mlp --trainer ppo       --config small --seed 42 --no-wandb
python train.py --policy mlp --trainer grpo      --config small --seed 42 --no-wandb
python train.py --policy mlp --trainer sac       --config small --seed 42 --no-wandb

# ActorCritic x compatible trainers

python train.py --policy ac  --trainer reinforce --config small --seed 42 --no-wandb
python train.py --policy ac  --trainer ppo       --config small --seed 42 --no-wandb

# GNN x compatible trainers

python train.py --policy gnn --trainer reinforce --config small --seed 42 --no-wandb
python train.py --policy gnn --trainer ppo       --config small --seed 42 --no-wandb
python train.py --policy gnn --trainer grpo      --config small --seed 42 --no-wandb
