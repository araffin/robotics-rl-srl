import pytest


def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true", default=False,
                     help="only run RL with ground truth, and quickly test all the envs")
    parser.addoption("--all", action="store_true", default=False,
                     help="run all the tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--fast") and config.getoption("--all"):
        raise AssertionError("Error: incompatible test flags requested")
    elif config.getoption("--fast"):
        skip_not_fast = pytest.mark.skip(reason="need to remove --fast option to run")
        for item in items:
            if "fast" not in item.keywords:
                item.add_marker(skip_not_fast)
    elif not config.getoption("--all"):
        skip_slow = pytest.mark.skip(reason="need to add --all option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
