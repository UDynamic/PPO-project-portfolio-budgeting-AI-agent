# src/env/__init__.py
#
# Registers PortfolioBudgeting-v0 with the Gymnasium registry so that
#
#     gym.make("PortfolioBudgeting-v0", config=cfg)
#
# works from anywhere that has src/ on sys.path.
#
# The entry_point uses the dotted module path relative to the package root.
# Because this file lives at src/env/__init__.py, and env.py is the sibling
# module src/env/env.py, the entry_point is "env.env:PortfolioBudgetingEnv".
#
# Callers must pass `config` as a keyword argument to gym.make() — it is
# a required __init__ parameter with no default.

import gymnasium as gym

gym.register(
    id      = "PortfolioBudgeting-v0",
    entry_point = "env.env:PortfolioBudgetingEnv",
    # kwargs are forwarded to __init__ on every gym.make() call.
    # config must be supplied by the caller; no default is provided here.
    kwargs  = {},
)