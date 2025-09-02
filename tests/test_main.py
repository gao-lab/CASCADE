from os import chdir, getcwd

from pytest import raises

from cascade.main import main


def test_cascade(adata, tmp_path):
    adata.write(tmp_path / "data.h5ad")

    main(
        [
            "discover",
            "-d",
            (tmp_path / "data.h5ad").as_posix(),
            "-m",
            (tmp_path / "discover.pt").as_posix(),
            "-i",
            (tmp_path / "info_discover.yaml").as_posix(),
            "--interv-key",
            "interv",
            "--use-size",
            "size",
            "--latent-dim",
            "4",
            "--latent-mod",
            "EmbLatent",
            "--max-epochs",
            "20",
            "--log-dir",
            (tmp_path / "log_dir").as_posix(),
            "-v",
        ]
    )
    assert (tmp_path / "discover.pt").exists()
    assert (tmp_path / "info_discover.yaml").exists()

    main(
        [
            "acyclify",
            "-m",
            (tmp_path / "discover.pt").as_posix(),
            "-g",
            (tmp_path / "acyc.gml.gz").as_posix(),
            "-i",
            (tmp_path / "info_acyclify.yaml").as_posix(),
            "-v",
        ]
    )
    assert (tmp_path / "acyc.gml.gz").exists()
    assert (tmp_path / "info_acyclify.yaml").exists()

    main(
        [
            "tune",
            "-d",
            (tmp_path / "data.h5ad").as_posix(),
            "-g",
            (tmp_path / "acyc.gml.gz").as_posix(),
            "-m",
            (tmp_path / "discover.pt").as_posix(),
            "-o",
            (tmp_path / "tune.pt").as_posix(),
            "-i",
            (tmp_path / "info_tune.yaml").as_posix(),
            "--interv-key",
            "interv",
            "--use-size",
            "size",
            "--max-epochs",
            "20",
            "-v",
        ]
    )
    assert (tmp_path / "tune.pt").exists()
    assert (tmp_path / "info_tune.yaml").exists()

    main(
        [
            "design",
            "-d",
            (tmp_path / "data.h5ad").as_posix(),
            "-m",
            (tmp_path / "tune.pt").as_posix(),
            "-t",
            (tmp_path / "data.h5ad").as_posix(),
            "-o",
            (tmp_path / "design.csv").as_posix(),
            "-u",
            (tmp_path / "design.pt").as_posix(),
            "-i",
            (tmp_path / "info_design.yaml").as_posix(),
            "--use-size",
            "size",
            "--design-scale-bias",
            "--max-epochs",
            "20",
            "-v",
        ]
    )
    assert (tmp_path / "design.csv").exists()
    assert (tmp_path / "design.pt").exists()
    assert (tmp_path / "info_design.yaml").exists()

    main(
        [
            "design_brute_force",
            "-d",
            (tmp_path / "data.h5ad").as_posix(),
            "-m",
            (tmp_path / "tune.pt").as_posix(),
            "-t",
            (tmp_path / "data.h5ad").as_posix(),
            "-o",
            (tmp_path / "design_brute_force.csv").as_posix(),
            "-p",
            (tmp_path / "pred.h5ad").as_posix(),
            "-i",
            (tmp_path / "info_design_brute_force.yaml").as_posix(),
            "--use-size",
            "size",
            "-k",
            "10",
        ]
    )
    assert (tmp_path / "design_brute_force.csv").exists()
    assert (tmp_path / "pred.h5ad").exists()
    assert (tmp_path / "info_design_brute_force.yaml").exists()

    main(
        [
            "counterfactual",
            "-d",
            (tmp_path / "data.h5ad").as_posix(),
            "-m",
            (tmp_path / "tune.pt").as_posix(),
            "-u",
            (tmp_path / "design.pt").as_posix(),
            "-p",
            (tmp_path / "ctfact.h5ad").as_posix(),
            "-i",
            (tmp_path / "info_ctfact.yaml").as_posix(),
            "--interv-key",
            "ctfact",
            "--use-size",
            "size",
            "-v",
        ]
    )
    assert (tmp_path / "ctfact.h5ad").exists()
    assert (tmp_path / "info_ctfact.yaml").exists()

    main(
        [
            "upgrade",
            "-m",
            (tmp_path / "tune.pt").as_posix(),
        ]
    )
    assert (tmp_path / "tune.pt").exists()


def test_devmgr(tmp_path):
    cwd = getcwd()
    chdir(tmp_path)
    main(["devmgr", "init", "--n-devices", "4"])
    acquired = main(["devmgr", "acquire", "--n-devices", "2"])
    with raises(ValueError):
        main(["devmgr", "acquire", "--n-devices", "3"])
    main(["devmgr", "release", "--devices", acquired.split(",")[0]])
    main(["devmgr", "acquire", "--n-devices", "3"])
    chdir(cwd)
