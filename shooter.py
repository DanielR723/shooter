import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize



# -------------------- Variables --------------------

# Simulation Variables
shooter_height = 0.390 # Height of the shooter (m)
target_height = 1.83 # Height of the target (m)
angle_range = np.pi / 4 # Tolerance for vertical angle when hitting the target (rad)
t_max = 10.0 # Max time to integrate to (s)
rpm_density = 10 # Number of values used when simulating rpms
angle_density = 10 # Number of values used when simulating angles
speed_density = 10 # Number of values used when simulating speeds
dir_density = 10 # Number of values used when simulating directions

# Shooter Constraints
rpm_min = 100 # Minimum rpm (2pi*rad/min)
rpm_max = 200 # Maximum rpm (2pi*rad/min)
angle_min = 0 # Minimum shooter angle (rad)
angle_max = np.pi / 2.0 # Maximum shooter angle (rad)
speed_min = 0.0 # Minimum bot speed (m/s)
speed_max = 100.0 # Maximum bot speed (m/s)
dir_min = 0.0 # Minimum bot angle (rad)
dir_max = 2.0 * np.pi # Maximum bot angle (rad)

# Ball Constants
m = 0.215 # Mass of the ball (kg)
R = 0.0750 # Radius of the ball (m)
A = np.pi * R**2 # Cross-sectional area of the ball (m^2)
C_d = 1 # Drag coefficient (constant)
k = 1 # S -> C_l approximate ratio (constant)

# Global Constants
g = 9.81 # Gravitational acceleration (m/s^2)
p = 1.14 # Air density (kg/m^3)

# Generate set of values to simulate
rpm = np.linspace(rpm_min, rpm_max, rpm_density) # RPM's (2pi*rad/min)
angle = np.linspace(angle_min, angle_max, angle_density) # Angle's (rad)
speed = np.linspace(speed_min, speed_max, speed_density) # Speeds the bot is traveling (m/s)
dir = np.linspace(dir_min, dir_max, dir_density) # Directions the bot is traveling (rad)

# -------------------- Variables --------------------



# -------------------- Data --------------------

speed_deg = 3 # Degree of speed polynomial
spin_deg = 3 # Degree of spin polynomial

speed_c = [] # Coefficients for speed polynomial (NEEDS TO BE INITIALIZED BEFORE DATATABLE)
spin_c = [] # Coefficients for spin polynomial (NEEDS TO BE INITIALIZED BEFORE DATATABLE)

test_rpm = np.array([100, 120, 140, 160, 180, 200]) # RPM's (2pi*rad/min)
test_speed = np.array([10, 20, 40, 65, 70, 100]) # Speed exiting shooter (m/s)
test_spin = np.array([100, 140, 145, 300, 325, 350]) # Spin exiting shooter (2pi*rad/s)

timestep = 0.09 # Timestep between snapshots (s)
test_y = -np.array([[0.0, 0.08, 0.18, 0.42, 0.78]]) # Y positions each frame (m)
t = np.arange(0, test_y.shape[1]) * timestep # Time for each snapshot (s)

# -------------------- Data --------------------



# -------------------- Setup --------------------

def rpm_to_params(rpm): # Convert RPM to speed and spin
    speed = np.zeros(rpm.shape)
    spin = np.zeros(rpm.shape)
    for i in range(speed_deg + 1):
        speed += speed_c[i] * rpm**(speed_deg - i)
    for i in range(spin_deg + 1):
        spin += spin_c[i] * rpm**(spin_deg - i)

    return speed, spin

def ode(t, y): # ODE's for integrator
    v = y[3:6]

    # Helper values
    speed = np.linalg.norm(v)
    if speed == 0:
        speed = 1e-8

    S = y[6] * R / speed
    C_l = k * S # APPROXIMATION

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

def ode_c(t, y, C_d): # ODE's for integrator
    v = y[3:6]

    # Helper values
    speed = np.linalg.norm(v)
    if speed == 0:
        speed = 1e-8

    S = y[6] * R / speed
    C_l = k * S # APPROXIMATION

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

def simulate(s, d, r, a): # Get the landing x of a certain rpm and angle
    speed, spin = rpm_to_params(r)

    # Set up integrator
    y0 = np.array([0.0, 0.0, shooter_height, s * np.cos(d) + speed * np.cos(a), s * np.sin(d), speed * np.sin(a), spin])
    sol = solve_ivp(ode, t_span=(0.0, t_max), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10)

    # Get return value
    if sol.t_events[0].size > 0:
        y_hit = sol.y_events[0][0]

        vertical = -y_hit[5] / np.linalg.norm(y_hit[3:6]) > np.cos(angle_range)
        if vertical:
            shot_dist = np.sqrt(y_hit[0]**2 + y_hit[1]**2)
            shot_dir = np.arctan2(y_hit[1], y_hit[0])

            return shot_dist, shot_dir
        else:
            return None, None
    else:
        return None, None

# -------------------- Setup --------------------



# -------------------- Calculation --------------------

def fit_shot_data(): # Fit a polynomial to data collected for speed and spin
    global speed_c, spin_c

    speed_c = np.polyfit(test_rpm, test_speed, speed_deg)
    spin_c = np.polyfit(test_rpm, test_spin, spin_deg)

def error(x):
    C_d = x[0]

    # Set up integrator
    y0 = np.zeros(7)
    sol = solve_ivp(ode_c, args=(C_d,), t_span=(0.0, t[-1]), y0=y0, t_eval=t, rtol=1e-8, atol=1e-10)
    est_y = sol.y[2]

    # Calculate error between estimation and data
    e = 0
    for row in test_y:
        e += np.sum((row - est_y)**2)
    
    return e

def optomize_C_d(): # Optomize C_d using test data
    global C_d

    x0 = [C_d]
    result = minimize(error, x0=x0, bounds=[(0.0, np.inf)])
    C_d = result.x[0]

# -------------------- Calculation --------------------



# -------------------- Lookup --------------------

def gen_lookup(): # Create the lookup table
    # Percentage visual
    percent_ratio = rpm_density * angle_density * speed_density * dir_density
    current_iter = 0
    percent_done = -1

    # Integrate
    datatable = []
    for i in range(speed_density):
        for j in range(dir_density):
            for k in range(rpm_density):
                for l in range(angle_density):
                    # Show perentage
                    current_iter += 1
                    percent = int(current_iter / percent_ratio * 100)
                    if percent != percent_done:
                        bar = int(current_iter / percent_ratio * 20)
                        print(f'\rIntegrating [{'-' * bar}{' ' * (20 - bar)}] %{percent}', end='')
                        percent_done = percent

                    # Populate table
                    s = speed[i]
                    d = dir[j]
                    r = rpm[k]
                    a = angle[l]

                    shot_dist, shot_dir = simulate(s, d, r, a)
                    if shot_dist:
                        datatable.append([shot_dist, shot_dir * 180 / np.pi, s, d, r, a * 180 / np.pi])

    print(f'\nForming lookup table with {len(datatable)} datapoints')

    # Sort table
    datatable = np.array(datatable)
    sorted = datatable[:, 0].argsort()

    # Write to CSV
    with open('shooter-lookup.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['landing distance', 'landing direction', 'bot speed', 'bot direction', 'rpm', 'angle'])
        for i in sorted:
            row = datatable[i, :]
            writer.writerow(row)

# -------------------- Lookup --------------------



# -------------------- Graph --------------------

def sim_path(s, d, r, a):
    speed, spin = rpm_to_params(np.array([r]))
    speed = speed[0]
    spin = spin[0]

    y0 = np.array([0.0, 0.0, shooter_height, s * np.cos(d) + speed * np.cos(a), s * np.sin(d), speed * np.sin(a), spin])
    sol = solve_ivp(ode, t_span=(0, t_max), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0, :], sol.y[1, :], sol.y[2, :])
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

def plot_shot_data(): # Plot the collected speed and spin datapoints against the polynomial curves
    x = np.linspace(rpm_min, rpm_max, rpm_density)
    y1, y2 = rpm_to_params(x)

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.scatter(test_rpm, test_speed)
    plt.scatter(test_rpm, test_spin)

    plt.show()

def plot_path_data(): # Plot predicted path compared to datapoints
    x = np.linspace(0.0, t[-1], 100)

    y0 = np.zeros(7)
    sol = solve_ivp(ode_c, args=(C_d,), t_span=(0.0, t[-1]), y0=y0, t_eval=x, rtol=1e-8, atol=1e-10)

    y = sol.y[2]

    plt.plot(x, y)
    for row in test_y:
        plt.scatter(t, row)

    plt.show()

# -------------------- Graph --------------------



fit_shot_data()
# optomize_C_d()
s, d, r, a = 33.33333333333333,4.886921905584122,188.88888888888889,50.0
sim_path(s, d, r, a * np.pi / 180)