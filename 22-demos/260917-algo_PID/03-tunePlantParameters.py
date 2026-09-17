import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# ---------------------------------------------------------------
# 0. Disturbance generator
# ---------------------------------------------------------------
seed = None #this will make each time an random seed instead of fixed 43
def make_disturbance(n, dt, kind="random_walk", seed=seed):
    rng = np.random.default_rng(seed)

    if kind == "constant":
        return np.full(n, 0.5)

    elif kind == "white":                       # high-freq noise
        return 0.5 + 0.3 * rng.standard_normal(n)

    elif kind == "random_walk":                 # low-freq drift
        d = np.zeros(n)
        d[0] = 0.5
        for k in range(1, n):
            d[k] = d[k-1] + 0.1 * rng.standard_normal() * np.sqrt(dt)
        return d

    elif kind == "sinusoidal":                  # periodic
        t = np.arange(n) * dt
        return 0.5 + 0.5 * np.sin(2*np.pi*0.5*t + rng.uniform(0, 2*np.pi))

    else:
        raise ValueError(kind)


# ---------------------------------------------------------------
# 1. Plant simulation
# ---------------------------------------------------------------
def simulate(Kp, Ki, Kd, disturbance, T=20.0, dt=0.01, r=1.0, c=0.2):
    """Run one PID simulation. `disturbance` must be an array of length int(T/dt)."""
    n = int(T / dt)
    t = np.linspace(0, T, n, endpoint=False)

    y = 0.0
    v = 0.0
    integral = 0.0
    prev_e = r - y

    ys = np.zeros(n)
    us = np.zeros(n)

    for k in range(n):
        e = r - y

        P = Kp * e
        integral += e * dt
        I = Ki * integral
        derivative = (e - prev_e) / dt
        D = Kd * derivative

        u = P + I + D

        # ✅ FIX 1: use disturbance[k], not the whole array
        v += (-c * v + u + disturbance[k]) * dt
        y += v * dt

        ys[k] = y
        us[k] = u
        prev_e = e

    return t, ys, us


# ---------------------------------------------------------------
# 2. Cost function — takes disturbance as an argument now
# ---------------------------------------------------------------
def cost(Kp, Ki, Kd, disturbance, r=1.0, T=20.0, dt=0.01,
         w_overshoot=2.0, w_effort=0.001):
    """Lower cost = better. Combines ITAE, overshoot, and control effort."""
    t, y, u = simulate(Kp, Ki, Kd, disturbance, T=T, dt=dt, r=r)
    e = r - y

    itae = np.sum(t * np.abs(e)) * dt
    overshoot = max(0.0, y.max() - r) * 100
    effort = np.sum(u ** 2) * dt

    return itae + w_overshoot * overshoot + w_effort * effort


# ---------------------------------------------------------------
# 3. Grid search
# ---------------------------------------------------------------
def tune_grid(Kp_range, Ki_range, Kd_range, disturbance, verbose=False):
    """Brute-force search. Uses the SAME disturbance for every candidate."""
    best = (None, None, None, float("inf"))

    for Kp, Ki, Kd in product(Kp_range, Ki_range, Kd_range):
        if Kp <= 0 or Ki < 0 or Kd < 0:
            continue
        if Ki >= Kp * (0.2 + Kd):   # Routh-Hurwitz stability for this plant
            continue

        c = cost(Kp, Ki, Kd, disturbance)
        if c < best[3]:
            best = (Kp, Ki, Kd, c)
            if verbose:
                print(f"new best: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}, cost={c:.3f}")

    return best


# ---------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------
if __name__ == "__main__":
    T = 20.0
    dt = 0.01
    n = int(T / dt)

    # Choose a disturbance kind: "constant", "white", "random_walk", "sinusoidal"
    kind = "random_walk"
    disturbance = make_disturbance(n, dt, kind=kind, seed=43)

    Kp_range = np.arange(0.5, 8.1, 0.5)
    Ki_range = np.arange(0.0, 2.1, 0.1)
    Kd_range = np.arange(0.0, 4.1, 0.5)

    best_Kp, best_Ki, best_Kd, best_cost = tune_grid(
        Kp_range, Ki_range, Kd_range, disturbance
    )

    print("\nBest gains:")
    print(f"  Kp = {best_Kp}")
    print(f"  Ki = {best_Ki}")
    print(f"  Kd = {best_Kd}")
    print(f"  cost = {best_cost:.3f}")

    # Plot best vs manual gains
    plt.figure(figsize=(10, 6))
    for label, (Kp, Ki, Kd) in [
        ("yours (1.0, 0.1, 0.3)", (1.0, 0.1, 0.3)),
        ("mine (4.0, 0.5, 1.5)", (4.0, 0.5, 1.5)),
        (f"tuned ({best_Kp}, {best_Ki}, {best_Kd})", (best_Kp, best_Ki, best_Kd)),
    ]:
        # ✅ FIX 2: pass disturbance here
        t, y, _ = simulate(Kp, Ki, Kd, disturbance, T=T, dt=dt)
        plt.plot(t, y, label=label)

    plt.axhline(1.0, color="k", linestyle="--", label="setpoint")
    plt.xlabel("time (s)")
    plt.ylabel("output y")
    plt.title(f"Auto-tuned vs manual gains  (disturbance = {kind})")
    plt.grid(True)
    plt.legend()
    plt.show()
