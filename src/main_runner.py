import os
from argparse import ArgumentParser

from vtab.runner import Runner


def parse_args():
    parser = ArgumentParser(description="Sequentially runs all config files in the work directory")
    parser.add_argument(
        "--work_area",
        type=str,
        default="./work",
        help="work area (where to put the yamls for the generated runs)",
    )
    parser.add_argument("--devices", type=str)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--idling_tolerance", type=int, default=10)
    return vars(parser.parse_args())


def main(work_area, devices, accelerator, num_workers, idling_tolerance):
    # set devices in a SLURM environment
    if devices is None:
        assert "SLURM_JOB_ID" in os.environ, f"use --devices when not using SLURM"
        assert "SLURM_LOCALID" in os.environ, "no SLURM_LOCALID found -> is required to infer the GPU index"
        devices = os.environ["SLURM_LOCALID"]
        idling_tolerance = 0
    Runner(work_area=work_area).run(
        devices=devices,
        accelerator=accelerator,
        num_workers=num_workers,
        idling_tolerance=idling_tolerance,
    )


if __name__ == '__main__':
    main(**parse_args())
