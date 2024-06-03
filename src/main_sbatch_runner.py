import os
import platform
import shlex
import sys
import uuid
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import yaml
import kappaconfig as kc


def get_parser():
    parser = ArgumentParser()
    # how many GPUs
    parser.add_argument("--gpus", type=int, default=1)
    #
    parser.add_argument("--time", type=str, required=True)
    parser.add_argument("--account", type=str)
    parser.add_argument("--qos", type=str)
    #
    return parser


def main(gpus, time, account, qos):
    # load template submit script
    with open("template_sbatch.sh") as f:
        template = f.read()

    if os.name == "nt":
        os.environ["HOME"] = "~/Documents/"

    # load config
    config = kc.DefaultResolver().resolve(kc.from_file_uri("sbatch_config.yaml"))
    # check paths exist
    chdir = Path(config["chdir"]).expanduser()
    assert chdir.exists(), f"chdir {chdir} doesn't exist"
    # default account
    account = account or config.get("default_account", None)
    # not every server has qos and qos doesnt need to be defined via CLI args
    qos = qos or config.get("default_qos")

    # get sbatch-only arguments
    parser = get_parser()
    args_to_filter = []
    # noinspection PyProtectedMember
    for action in parser._actions:
        if action.dest == "help":
            continue
        # currently only supports to filter out args with -- prefix
        assert len(action.option_strings) == 1
        assert action.option_strings[0].startswith("--")
        args_to_filter.append(action.option_strings[0])
    # filter out sbatch-only arguments
    train_args = []
    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1 + i]
        if arg.startswith("--") and arg in args_to_filter:
            i += 2
        else:
            train_args.append(arg)
            i += 1
    cli_args_str = " ".join(map(shlex.quote, train_args))

    # patch template
    patched_template = template.format(
        time=time,
        gpus=gpus,
        account=account,
        qos=qos,
        cli_args=cli_args_str,
        **config,
    )
    print(patched_template)

    # create a shell script
    out_path = Path("submit")
    out_path.mkdir(exist_ok=True)
    fname = f"{datetime.now():%m.%d-%H.%M.%S}-{uuid.uuid4()}.sh"
    with open(out_path / fname, "w") as f:
        f.write(patched_template)

    # execute the shell script
    if os.name != "nt":
        os.system(f"sbatch submit/{fname}")


if __name__ == "__main__":
    main(**vars(get_parser().parse_known_args()[0]))
