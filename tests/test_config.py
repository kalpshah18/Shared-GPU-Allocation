"""tests/test_config.py — validation, derived fields and YAML loading of sim.config.Config."""

import math

import pytest
import yaml

from sim.config import Config


def test_defaults_match_proposal_base_configuration():
    c = Config()
    assert (c.n, c.k, c.T, c.rho, c.v_max, c.n_seeds, c.n_bootstrap) == (50, 10, 1000, 0.25, 1.0, 30, 10_000)
    assert c.valuation_dist == "uniform" and c.alpha == 0.0 and c.lambda_ == 1.0


@pytest.mark.parametrize("n,k", [(10, 3), (50, 10), (7, 2), (2, 1)])
def test_delta_is_twice_the_round_robin_cycle(n, k):
    assert Config(n=n, k=k).delta == 2 * math.ceil(n / k)


def test_delta_multiplier():
    assert Config(n=10, k=5, delta_multiplier=3).delta == 3 * 2


@pytest.mark.parametrize("kwargs", [
    dict(n=5, k=5), dict(n=5, k=6), dict(n=5, k=0), dict(n=1, k=1),
    dict(T=0), dict(v_max=0.0), dict(v_max=-1.0),
    dict(rho=-0.1), dict(rho=1.5),
    dict(lambda_=-0.5), dict(c=0.5),
    dict(alpha=-0.1), dict(alpha=1.0),
    dict(valuation_dist="gaussian"), dict(ar1_mode="bogus"), dict(n_focal=0),
])
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_rho_endpoints_allowed(rho):
    assert Config(rho=rho).rho == rho


def test_replace_revalidates_and_rederives_delta():
    c = Config(n=10, k=2)
    c2 = c.replace(k=5)
    assert c2.k == 5 and c2.delta == 2 * 2 and c.k == 2     # original untouched
    with pytest.raises(ValueError):
        c.replace(k=10)


def test_to_dict_contains_every_field_and_is_json_ready():
    import json
    d = Config(n=10, k=2).to_dict()
    assert d["n"] == 10 and d["delta"] == 2 * 5 and "lambda_" in d
    json.dumps(d)


def test_from_yaml_roundtrip_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"n": 20, "k": 4, "T": 30, "lambda_": 2.0,
                                    "delta": 999, "not_a_field": 1}))
    c = Config.from_yaml(path, rho=0.5)
    assert (c.n, c.k, c.T, c.lambda_, c.rho) == (20, 4, 30, 2.0, 0.5)
    assert c.delta == 2 * 5                                 # derived, not taken from the file


def test_base_yaml_loads_to_the_default_configuration():
    from pathlib import Path
    base = Config.from_yaml(Path(__file__).resolve().parent.parent / "config" / "base.yaml")
    assert base.to_dict() == Config().to_dict()
