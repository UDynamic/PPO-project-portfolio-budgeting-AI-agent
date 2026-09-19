# src/tests/env_test_configs/plan_calculator.py


# ⚠️ refine this one duration ambiguity


import math


def beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    dx = x / 200
    total = sum(((k + 0.5) * dx) ** (a - 1) * (1 - (k + 0.5) * dx) ** (b - 1) * dx
                for k in range(200))
    B = math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    return min(1.0, total / B)


def plan_values(duration: int, a: float = 6.0, b: float = 6.0) -> list[float]:
    """
    Returns cumulative planned progress fractions, one per timestep action.

    A project with `duration` has (duration + 1) periods and (duration + 1) plan values,
    evaluated at the end of each period: t/n for t in 1 .. n+1, where n = duration + 1.

      duration=0 -> 1 value  -> [beta_cdf(1/1)] = [1.0]
      duration=1 -> 2 values -> [beta_cdf(1/2), beta_cdf(2/2)]
      duration=2 -> 3 values -> [beta_cdf(1/3), beta_cdf(2/3), beta_cdf(3/3)]
    """
    if duration < 0:
        raise ValueError(f"duration must be >= 0, got {duration}")
    n = duration + 1
    return [beta_cdf(t / n, a, b) for t in range(1, n + 1)]


if __name__ == "__main__":
    duration = int(input("Enter project duration: "))
    values = plan_values(duration)
    print(f"\nPlan values for duration={duration} (a=6, b=6):")
    for t, v in enumerate(values):
        print(f"  timestep {t}: {v:.4f}")