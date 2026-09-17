import numpy as np
import matplotlib.pyplot as plt

def simulate(controller, T=20.0, dt=0.01):
    n = int(T / dt)
    t = np.linspace(0, T, n, endpoint=False)

    # Plant state
    y = 0.0      # position / output
    v = 0.0      # velocity
    r = 1.0      # setpoint

    # Controller state
    integral = 0.0
    prev_e = r - y

    # Plant parameters
    c = 0.2      # weak friction / damping
    d = 0.5      # constant disturbance

    # PID gains
    Kp = 1 
    Ki = 0.1
    Kd = 0.3

    ys = np.zeros(n)

    for k in range(n):
        e = r - y

        # PID terms
        P = Kp * e
        integral += e * dt
        I = Ki * integral
        derivative = (e - prev_e) / dt
        D = Kd * derivative

        if controller == "P":
            u = P
        elif controller == "PI":
            u = P + I
        elif controller == "PD":
            u = P + D
        elif controller == "PID":
            u = P + I + D
        else:
            raise ValueError("unknown controller")

        # Plant dynamics:
        # mass = 1
        # v' = -c*v + u + d
        # y' = v
        v += (-c * v + u + d) * dt
        y += v * dt

        ys[k] = y
        prev_e = e

    return t, ys


controllers = ["P", "PI", "PD", "PID"]
y_controllers={}

plt.figure(figsize=(10, 6))
for name in controllers:
    t, y = simulate(name, 50)
    y_controllers[name] = y
    plt.plot(t, y, label=name)

plt.axhline(1.0, color="k", linestyle="--", label="setpoint")
plt.xlabel("time (s)")
plt.ylabel("output y")
plt.title("Why P alone is not enough: constant disturbance d=0.5")
plt.grid(True)
plt.legend()
plt.show()

for name,y in y_controllers.items():
    print(f"{name:>3}: final y = {y[-1]:.4f}, error = {1 - y[-1]:.4f}")
