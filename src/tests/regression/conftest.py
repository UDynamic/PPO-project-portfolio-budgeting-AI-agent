# tests/regression/conftest.py
#
# pytest configuration for the regression suite.
# No fixtures defined here — all parametrization is in harness.py.
#
# To run:
#   pytest tests/regression/                        # all fixtures
#   pytest tests/regression/ -k "TC-10SF-R"         # one config
#   pytest tests/regression/ -k "optimal"           # one method across all configs
#   pytest tests/regression/ -v                     # verbose (show fixture IDs)
#   pytest tests/regression/ --tb=short             # compact tracebacks