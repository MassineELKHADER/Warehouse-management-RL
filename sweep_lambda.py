"""
Lambda sensitivity sweep — runs PPO for each lambda value sequentially.

Usage:
    python sweep_lambda.py                          # all lambdas, seed 42
    python sweep_lambda.py --seeds 42 123 7         # 3 seeds per lambda
    python sweep_lambda.py --lambdas 1 5 10 20      # custom lambda values
    python sweep_lambda.py --no-wandb               # skip WandB (dry run)
"""

import argparse
import subprocess
import sys
import time

LAMBDAS_DEFAULT = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas",    type=float, nargs="+", default=LAMBDAS_DEFAULT)
    parser.add_argument("--seeds",      type=int,   nargs="+", default=[42])
    parser.add_argument("--config",     default="small")
    parser.add_argument("--timesteps",  type=int,   default=1_000_000)
    parser.add_argument("--n-envs",     type=int,   default=14)
    parser.add_argument("--no-wandb",   action="store_true")
    args = parser.parse_args()

    total_runs = len(args.lambdas) * len(args.seeds)
    print(f"Lambda sweep: {len(args.lambdas)} lambdas × {len(args.seeds)} seeds = {total_runs} runs")
    print(f"Lambdas: {args.lambdas}")
    print(f"Seeds:   {args.seeds}")
    print(f"Config:  {args.config}, Timesteps: {args.timesteps:,}, Envs: {args.n_envs}")
    print("-" * 60)

    failed = []
    for i, lam in enumerate(args.lambdas):
        for seed in args.seeds:
            run_id = f"lambda={lam}, seed={seed}"
            print(f"\n[{i+1}/{len(args.lambdas)}] Starting {run_id}")
            t0 = time.time()

            cmd = [
                sys.executable, "train_sb3.py",
                "--algo",           "ppo",
                "--config",         args.config,
                "--seed",           str(seed),
                "--timesteps",      str(args.timesteps),
                "--n-envs",         str(args.n_envs),
                "--lambda-penalty", str(lam),
            ]
            if args.no_wandb:
                cmd.append("--no-wandb")

            result = subprocess.run(cmd)
            elapsed = time.time() - t0

            if result.returncode != 0:
                print(f"  FAILED ({run_id}) after {elapsed:.0f}s")
                failed.append(run_id)
            else:
                print(f"  Done ({run_id}) in {elapsed/60:.1f} min")

    print("\n" + "=" * 60)
    print(f"Sweep complete. {total_runs - len(failed)}/{total_runs} runs succeeded.")
    if failed:
        print(f"Failed runs: {failed}")


if __name__ == "__main__":
    main()
