import pytest


def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true", default=False,
                     help="only run RL with ground truth, and quickly test all the envs")
    parser.addoption("--exaustive", action="store_true", default=False,
                     help="run all the tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--fast") and config.getoption("--exaustive"):
        raise AssertionError("Error: incompatible test flags requested")
    elif config.getoption("--fast"):
        skip_gpu = pytest.mark.skip(reason="need to remove --fast option to run")
        for item in items:
            if "fast" not in item.keywords:
                item.add_marker(skip_gpu)
    elif not config.getoption("--exaustive"):
        skip_gpu = pytest.mark.skip(reason="need to add --exaustive option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_gpu)
