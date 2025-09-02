from os.path import join, exists
from functools import reduce
from operator import add
from random import choices, seed
from collections import defaultdict


def conf_pattern(conf, name=None):
    if not isinstance(conf, dict):
        return f"{{{name}}}"
    return "-".join(f"{k}={{{k}}}" for k in conf)


def conf_values(conf, name=None):
    if not isinstance(conf, dict):
        return {name: conf}
    return {k: list(range(v)) if k.endswith("sd") else v for k, v in conf.items()}


def conf_values_zip(conf):
    defaults = {k: v["default"] if isinstance(v, dict) else v for k, v in conf.items()}
    zip_map = defaultdict(list)
    for k, v in conf.items():
        if isinstance(v, dict):
            zip_map[k].extend(v["choices"])
            for k_ in defaults:
                if k_ != k:
                    zip_map[k_].extend([defaults[k_]] * len(v["choices"]))
    return zip_map


def target_files(directories, files):
    def per_directory(directory):
        if exists(join(directory, ".blacklist")):
            return []
        if isinstance(files, list):
            return [join(directory, file) for file in files]
        else:  # Function
            return [join(directory, file) for file in files(directory)]

    return reduce(add, map(per_directory, directories))


def random_expand(n, s=0):
    seed(s)

    def _random_expand(pattern, **kws):
        return expand(
            pattern,
            zip,
            **{
                key: choices(val, k=n) if isinstance(val, list) else [val] * n
                for key, val in kws.items()
            },
        )

    return _random_expand
