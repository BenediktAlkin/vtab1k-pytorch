import os


def change_fname(fname, stage_name=None, paramstr=None):
    split = fname.split("__")
    if stage_name is not None:
        split[2] = stage_name
    if paramstr is not None:
        split[3] = paramstr
    return "__".join(split)


def fname_to_paramstr(fname):
    fname = fname.replace(".yaml", "")
    split = fname.split("__")
    return split[3]


def remove_seed_from_paramstr(paramstr):
    split = paramstr.split("_")
    split = [item for item in split if not item.startswith("seed=")]
    return "_".join(split)


def fname_to_summarized_name(fname):
    # debug__5fizs74e__summarize__lr=0.0005_seed=0.yaml -> debug__5fizs74e
    return "__".join(fname.split("__")[:2])


def fname_to_dynamic_params(fname):
    # 'debug__5fizs74e__run_eval__lr=0.0005_seed=4.yaml' -> 'debug__5fizs74e__run_eval__lr=0.0005_seed=4'
    fname = fname.replace(".yaml", "")
    # 'debug__5fizs74e__run_eval__lr=0.0005_seed=4' -> 'lr=0.0005_seed=4'
    dparams_str = fname.split("__")[3]
    # 'lr=0.0005_seed=4' -> ['lr=0.0005', 'seed=4']
    dparams_split = dparams_str.split("_")
    # ['lr=0.0005', 'seed=4'] -> dict(lr=0.0005, seed=4)
    dparams_dict = {}
    for dparam in dparams_split:
        name, value = dparam.split("=")
        dparams_dict[name] = value
    return dparams_dict


def get_pipeline_fnames(uri, pipeline_id, stage_name=None):
    fnames = []
    for fname in os.listdir(uri):
        split = fname.split("__")
        # check correct length (e.g. initial pipeline yamls dont need to have a pipeline id)
        if len(split) < 3:
            continue
        # check pipeline_id
        if split[1] != pipeline_id:
            continue
        # [OPTIONAL] check stage_name
        if stage_name is not None and split[2] != stage_name:
            continue
        fnames.append(fname)
    return fnames
