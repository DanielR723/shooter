import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- Variables ----------

# Simulation Variables
target_height = 2 # Height of the target (m)
angle_range = np.pi / 4 # Tolerance for vertical angle when hitting the target (rad)
t_max = 10 # Max time to integrate to (s)
rpm_density = 100 # Number of values used when simulating rpms
angle_density = 100 # Number ov values used when simulating angles

# Shooter Constraints
rpm_min = 100 # Minimum rpm (2pi*rad/min)
rpm_max = 200 # Maximum rpm (2pi*rad/min)
angle_min = 0 # Minimum shooter angle (rad)
angle_max = np.pi / 2 # Maximum shooter angle (rad)

# Ball Constants
m = 0.23 # Mass of the ball (kg)
R = 0.075 # Radius of the ball (m)
A = np.pi * R**2 # Cross-sectional area of the ball (m^2)
C_d = 1 # Drag coefficient (constant)
k = 1 # S -> C_l approximate ratio (constant)

# Global Constants
g = 9.81 # Gravitational acceleration (m/s^2)
p = 1.21 # Air density (kg/m^3)

# Generate set of values to simulate
rpm = np.linspace(rpm_min, rpm_max, num=rpm_density) # RPM's (2pi*rad/min)
angle = np.linspace(angle_min, angle_max, num=angle_density) # Angle's (rad)

# ---------- Variables ----------

# ---------- Integrator ----------

print('\nIntegrating')

def ode(t, y):
    v = y[3:6]

    # Helper values
    speed = np.linalg.norm(v)

    S = y[6] * R / speed
    C_l = k * S

    # Forces
    drag = -1 / 2 * p * C_d * A * speed * v
    magnus = 1 / 2 * p * C_l * A * speed**2 * np.cross(np.array([0, 1, 0]), v / speed)
    gravity = np.array([0, 0, -m * g])

    # Differentials
    dydt = np.zeros(7)
    dydt[0:3] = v
    dydt[3:6] = (drag + magnus + gravity) / m
    dydt[6] = 0

    return dydt

def hit_event(t, y):
    return y[2] - target_height

hit_event.direction = -1
hit_event.terminal = True

def simulate(rpm, angle):
    # PLACEHOLDER RPM FUNCTIONS
    speed = rpm / 10
    spin = 2 * np.pi / 60 * rpm

    y0 = np.array([0.0, 0.0, 0.0, speed * np.cos(angle), 0.0, speed * np.sin(angle), spin])
    sol = solve_ivp(ode, t_span=(0, t_max), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10)

    vertical = abs(np.arctan2(sol.y[3, -1], -sol.y[5, -1])) < angle_range
    if sol.t_events[0].size == 1 and vertical:
        return sol.y[0, -1]
    else:
        return None

datatable = []
for r in rpm:
    for a in angle:
        x = simulate(r, a)
        if x:
            datatable.append([x, r, a])

# ---------- Integrator ----------

# ---------- Lookup Table ----------

print(f'Forming lookup table with {len(datatable)} datapoints\n')

datatable = np.array(datatable)
sorted = datatable[:, 0].argsort()

with open('shooter-lookup.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['distance', 'rpm', 'angle'])
    for i in sorted:
        row = datatable[i, :]
        writer.writerow(row)

# ---------- Lookup Table ----------

# ---------- Graph ----------

rpm = 150.50505050505052
angle = 1.3803967720318788

speed = rpm / 10
spin = 2 * np.pi / 60 * rpm

y0 = np.array([0.0, 0.0, 0.0, speed * np.cos(angle), 0.0, speed * np.sin(angle), spin])
sol = solve_ivp(ode, t_span=(0, t_max), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10)

plt.plot(sol.y[0, :], sol.y[2, :])
plt.axis('equal')
plt.show()

# ---------- Graph ----------