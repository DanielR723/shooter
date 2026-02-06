import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize



# -------------------- Variables --------------------

# Simulation Variables
shooter_height = 1 # Height of the shooter (m)
target_height = 2 # Height of the target (m)
angle_range = np.pi / 4 # Tolerance for vertical angle when hitting the target (rad)
t_max = 10 # Max time to integrate to (s)
rpm_density = 25 # Number of values used when simulating rpms
angle_density = 25 # Number ov values used when simulating angles

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

# -------------------- Variables --------------------



# -------------------- Data --------------------

num_datapoints = 3 # Number of datapoints
fps = 4 # FPS of the camera (frames/s)
test_rpm = np.array([100, 140, 180]) # RPM's (2pi*rad/min)
test_angle = np.array([1.0, 1.5, 2.0]) # Angle's (rad)
test_x = np.array([[0.5, 1.0, 1.5, 2.0],
                   [1.2, 2.4, 3.0, 4.0],
                   [0.2, 0.5, 0.9, 1.5]]) # X Positions (m)
test_y = np.array([[0.2, 0.4, 0.6, 0.8],
                   [0.4, 0.8, 1.2, 1.6],
                   [0.6, 1.2, 2.0, 3.0]]) # Y Positions (m)
t = np.arange(0, test_x.shape[1]) / fps

# -------------------- Data --------------------



# -------------------- Setup --------------------

def rpm_to_data(rpm): # Convert RPM to speed and spin
    # PLACEHOLDER RPM FUNCTIONS
    speed = rpm / 10
    spin = 2 * np.pi / 60 * rpm

    return speed, spin

def ode(t, y): # ODE's for integrator
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

def hit_event(t, y): # Detect hits
    return y[2] - target_height

hit_event.direction = -1
hit_event.terminal = True

def simulate(rpm, angle): # Get the landing x of a certain rpm and angle
    speed, spin = rpm_to_data(rpm)

    # Set up integrator
    y0 = np.array([0.0, 0.0, shooter_height, speed * np.cos(angle), 0.0, speed * np.sin(angle), spin])
    sol = solve_ivp(ode, t_span=(0, t_max), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10)

    # Get return value
    vertical = abs(np.arctan2(sol.y[3, -1], -sol.y[5, -1])) < angle_range
    if sol.t_events[0].size == 1 and vertical:
        return sol.y[0, -1]
    else:
        return None

# -------------------- Setup --------------------


# -------------------- Lookup --------------------

def gen_lookup(): # Create the lookup table
    # Percentage visual
    rpm_len = rpm.shape[0]
    angle_len = angle.shape[0]
    percent_ratio = rpm_len * angle_len
    percent_done = -1

    # Integrate
    datatable = []
    for i in range(rpm_len):
        for j in range(angle_len):
            # Show perentage
            percent = int((i * angle_len + j + 1) / percent_ratio * 100)
            if percent != percent_done:
                bar = int((i * angle_len + j + 1) / percent_ratio * 20)
                print(f'\rIntegrating [{'-' * bar}{' ' * (20 - bar)}] %{percent}', end='')
                percent_done = percent

            # Populate table
            r = rpm[i]
            a = angle[j]

            x = simulate(r, a)
            if x:
                datatable.append([x, r, a])

    print(f'\n Forming lookup table with {len(datatable)} datapoints\n')

    # Sort table
    datatable = np.array(datatable)
    sorted = datatable[:, 0].argsort()

    # Write to CSV
    with open('shooter-lookup.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['distance', 'rpm', 'angle'])
        for i in sorted:
            row = datatable[i, :]
            writer.writerow(row)

# -------------------- Lookup --------------------



# -------------------- Calculator --------------------

def ode_c(t, y, C_d, k): # ODE's for integrator with constants input
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

def error(x0): # Get the error between simulation and test data
    C_d, k = x0

    e = 0
    for i in range(num_datapoints):
        speed, spin = rpm_to_data(test_rpm[i])
        angle = test_angle[i]
        x = test_x[i]
        y = test_y[i]

        y0 = np.array([x[0], 0.0, y[0], speed * np.cos(angle), 0.0, speed * np.sin(angle), spin])
        sol = solve_ivp(ode_c, t_span=(0, t[-1]), y0=y0, args=(C_d, k), t_eval=t, rtol=1e-8, atol=1e-10)

        pred_x = sol.y[0]
        pred_y = sol.y[2]

        e += np.sum((pred_x - x)**2 + (pred_y - y)**2)

    return e

def calc_constants(C_d_guess, k_guess): # Calculate the optimal C_d and k
    x0 = [C_d_guess, k_guess]
    result = minimize(error, x0=x0)
    return result.x

# -------------------- Calculator --------------------



# -------------------- Graph --------------------

def graph(rpm, angle):
    speed, spin = rpm_to_data(rpm)

    y0 = np.array([0.0, 0.0, shooter_height, speed * np.cos(angle), 0.0, speed * np.sin(angle), spin])
    sol = solve_ivp(ode, t_span=(0, t_max), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10)

    plt.plot(sol.y[0, :], sol.y[2, :])
    plt.axis('equal')
    plt.show()

# -------------------- Graph --------------------



print()
# gen_lookup()
graph(145.83333333333334, 1.3744467859455345)