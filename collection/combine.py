import argparse
import glob
import gzip
import os

import h5py
import wandb
from tqdm.auto import tqdm


def get_run(sweep_runs, ruleset_id):
    wandb_run = [r for r in sweep_runs if r.config["ruleset_id"] == ruleset_id]
    assert len(wandb_run) == 1
    return wandb_run[0]


def extract_id(filename):
    return int(os.path.basename(filename).split("-")[-1].split(".")[0])


def main(args):
    print("Processing sweep runs...")
    api = wandb.Api()
    all_runs = api.runs(args.wandb_entity)
    dataset_runs = [r for r in tqdm(all_runs) if hasattr(r.sweep, "id") and r.sweep.id == args.wandb_sweep]

    print("Combining...")
    files = glob.glob(os.path.join(args.data_path, "*.gz"))
    files = sorted(files, key=lambda f: extract_id(f))

    with h5py.File(args.combined_path, "w", rdcc_nbytes=5e9, rdcc_nslots=20000) as new_df:
        idx = 0
        for file in tqdm(files):
            try:
                with gzip.open(file, "rb") as gf:
                    with h5py.File(gf, "r") as df:
                        # checking that agent achieved return >= thrs, else skip
                        wandb_run = get_run(dataset_runs, df.attrs["ruleset-id"])

                        if "final_return" not in wandb_run.summary:
                            print(f"Corrupted run {file}, skipping...")
                            continue

                        if wandb_run.summary["final_return"] < args.final_return_thrs:
                            continue

                        assert str(idx) not in new_df.keys(), "key already exists"
                        g = new_df.create_group(str(idx))
                        g.attrs.update(df.attrs)

                        g.create_dataset(
                            "states",
                            shape=df["states"].shape,
                            dtype=df["states"].dtype,
                            data=df["states"][:],
                            compression="gzip",
                            compression_opts=6,
                            chunks=(1, 4096, 5, 5),
                        )
                        g.create_dataset(
                            "actions",
                            shape=df["actions"].shape,
                            dtype=df["actions"].dtype,
                            data=df["actions"][:],
                            compression="gzip",
                            compression_opts=6,
                            chunks=(1, 4096),
                        )
                        g.create_dataset(
                            "rewards",
                            shape=df["rewards"].shape,
                            dtype=df["rewards"].dtype,
                            data=df["rewards"][:],
                            compression="gzip",
                            compression_opts=6,
                            chunks=(1, 4096),
                        )
                        g.create_dataset(
                            "dones",
                            shape=df["dones"].shape,
                            dtype=df["dones"].dtype,
                            data=df["dones"][:],
                            compression="gzip",
                            compression_opts=6,
                            chunks=(1, 4096),
                        )
                        g.create_dataset(
                            "expert_actions",
                            shape=df["expert_actions"].shape,
                            dtype=df["expert_actions"].dtype,
                            data=df["expert_actions"][:],
                            compression="gzip",
                            compression_opts=6,
                            chunks=(1, 4096),
                        )
            
            except OSError:
                print(f"Corrupted file {file}, skipping...")
                continue

            idx = idx + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-sweep", type=str)
    parser.add_argument("--final-return-thrs", type=float, default=0.3)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--combined-path", type=str)
    args = parser.parse_args()
    main(args)
