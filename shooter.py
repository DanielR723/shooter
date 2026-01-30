import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------- Variables ----------

# Simulation Variables
target_height = 2 # Height of the target (m)
max_height = 100 # Max height of the ball (m)
angle_range = np.pi / 4 # Tolerance for vertical angle when hitting the target (rad)
dt = 0.01 # Timestep used when integrating (s)
sim_density = 100 # Number of values used when simulating rpms or shooter angles, Note: Runs sim_density^2 simulations
plot_paths = False # DEBUG, plot paths of simulated trajectories
plot_density = False # DEBUG, plot the x coordinates of successfull simulations
plot_prediction = True # DEBUG, simulate and plot a predicted shot versus a naive prediction

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
p = 1.26 # Air density (kg/m^3)

# ---------- Variables ----------

# ---------- Setup ----------

print('\nSetup')

# Generate set of values to simulate
rpm = np.linspace(rpm_min, rpm_max, num=sim_density)
angle = np.linspace(angle_min, angle_max, num=sim_density)

# Flattened cartesian product
rpm = np.repeat(rpm, sim_density)
angle = np.tile(angle, sim_density)

# PLACEHOLDER RPM FUNCTIONS
speed = rpm / 10
spin = 2 * np.pi / 60 * rpm

# Helper value
N = sim_density**2

# Projectile setup
pos = np.zeros((N, 3)) # Posision (m)
v = np.column_stack((speed * np.cos(angle), np.zeros(N), speed * np.sin(angle))) # Velocity (m/s)
omega = np.column_stack((np.zeros(N), spin, np.zeros(N))) # Spin speed (rad/s)

# ---------- Setup ----------

# ---------- Integrator ----------

print('Integrating')

# Simulation information
running = np.full(N, True)
hit = np.full(N, False)

x = np.empty((0, N))
y = np.empty((0, N))

gravity = np.array([0, 0, -m * g]) # Gravitational force
while np.any(running):
    # Helper values
    current_speed = np.linalg.norm(v, axis=1)
    current_spin = np.linalg.norm(omega, axis=1)

    S = current_spin * R / current_speed
    C_l = k * S

    # Calculate forces
    drag = -1 / 2 * p * C_d * A * current_speed[running][:, None] * v[running]
    magnus = 1 / 2 * p * C_l[running][:, None] * A * current_speed[running][:, None]**2 * np.cross(omega[running] / current_spin[running][:, None], v[running] / current_speed[running][:, None])
    force = drag + magnus + gravity

    v[running] += force / m * dt
    last_pos = pos.copy()
    pos[running] += v[running] * dt

    # Collision logic
    below = pos[:, 2] < target_height
    above = pos[:, 2] > max_height
    downward = v[:, 2] < 0
    crossing = np.sign(last_pos[:, 2] - target_height) != np.sign(pos[:, 2] - target_height)
    in_range = np.abs(np.atan2(v[:, 0], -v[:, 2])) < angle_range

    running[(below & downward) | above] = False
    hit[downward & crossing & in_range] = True

    # Plotting
    if plot_paths:
        x = np.vstack((x, pos[:, 0]))
        y = np.vstack((y, pos[:, 2]))

if plot_paths:
    for i in range(x.shape[1]):
        plt.plot(x[:, i], y[:, i])

    plt.axis('equal')
    plt.show()

# ---------- Integrator ----------

# ---------- Lookup Table ----------

print(f'Forming lookup table with {np.sum(hit)} datapoints')

with open('shooter-lookup.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['distance', 'rpm', 'angle_rad'])
    for d, r, a in zip(pos[:, 0][hit], rpm[hit], angle[hit]):
        writer.writerow([d, r, a])

if plot_density:
    plt.cla()
    plt.scatter(pos[:, 0][hit], rpm[hit])
    plt.show()

# ---------- Lookup Table ----------

# ---------- Test ----------

if plot_prediction:
    print('Interpolating')

    from scipy.interpolate import RBFInterpolator

    X = np.column_stack((rpm[hit], angle[hit]))
    y = pos[:, 0][hit]
    f = RBFInterpolator(X, y)

    print('Running Test')

    target_x = 5.0

    from scipy.optimize import minimize

    def loss(u):
        return (f([u])[0] - target_x)**2

    res = minimize(
        loss,
        x0=[(rpm_min + rpm_max) / 2, (angle_min + angle_max) / 2],
        bounds=[(rpm_min, rpm_max), (angle_min, angle_max)]
    )

    pred_rpm, pred_angle = res.x

    # Naive approximation
    desired_angle = np.pi * 3 / 8
    naive_speed = np.sqrt(g * target_x**2 / (2 * np.cos(desired_angle)**2 * (target_x * np.tan(desired_angle) - target_height)))

    speed = np.array([pred_rpm / 10, naive_speed])
    spin = np.array([2 * np.pi / 60 * pred_rpm, 100])
    angle = np.array([pred_angle, desired_angle])

    # Helper value
    N = 2

    # Projectile setup
    pos = np.zeros((N, 3)) # Posision (m)
    v = np.column_stack((speed * np.cos(angle), np.zeros(N), speed * np.sin(angle))) # Velocity (m/s)
    omega = np.column_stack((np.zeros(N), spin, np.zeros(N))) # Spin speed (rad/s)

    # Simulation information
    running = np.full(N, True)

    x = np.empty((0, N))
    y = np.empty((0, N))

    gravity = np.array([0, 0, -m * g]) # Gravitational force
    while np.any(running):
        # Helper values
        current_speed = np.linalg.norm(v, axis=1)
        current_spin = np.linalg.norm(omega, axis=1)

        S = current_spin * R / current_speed
        C_l = k * S

        # Calculate forces
        drag = -1 / 2 * p * C_d * A * current_speed[running][:, None] * v[running]
        magnus = 1 / 2 * p * C_l[running][:, None] * A * current_speed[running][:, None]**2 * np.cross(omega[running] / current_spin[running][:, None], v[running] / current_speed[running][:, None])
        force = drag + magnus + gravity

        v[running] += force / m * dt
        last_pos = pos.copy()
        pos[running] += v[running] * dt

        # Collision logic
        below = pos[:, 2] < target_height
        downward = v[:, 2] < 0

        running[below & downward] = False

        # Plotting
        x = np.vstack((x, pos[:, 0]))
        y = np.vstack((y, pos[:, 2]))

    for i in range(x.shape[1]):
        plt.plot(x[:, i], y[:, i])

    plt.scatter(target_x, target_height)
    plt.axis('equal')
    plt.show()

# ---------- Test ----------