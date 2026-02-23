import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
# 44 -> 46 from horizontal - 2000/3000/4000/5000/6000 RPM
# 34 -> 56 from horizontal - 2000/3000/4000/5000/6000 RPM
# Sanity check: 39 -> 51 from horizontal - 3000 RPM



# -------------------- Variables --------------------

# Simulation Variables
shooter_height = 0.390 # Height of the shooter (m)
target_height = 1.83 # Height of the target (m)
angle_range = np.pi / 4 # Tolerance for vertical angle when hitting the target (rad)
t_max = 10.0 # Max time to integrate to (s)
speed_density = 1 # Number of values used when simulating speeds
dir_density = 1 # Number of values used when simulating directions
rpm_density = 100 # Number of values used when simulating rpms
angle_density = 100 # Number of values used when simulating angles

# Shooter Constraints
speed_min = 0.0 # Minimum bot speed (m/s)
speed_max = 5.1 # Maximum bot speed (m/s)
dir_min = 0.0 # Minimum bot angle (rad)
dir_max = 2.0 * np.pi # Maximum bot angle (rad)
rpm_min = 2000.0 # Minimum rpm (2pi*rad/min)
rpm_max = 6000.0 # Maximum rpm (2pi*rad/min)
angle_min = 46.0 * np.pi / 180.0 # Minimum shooter angle (rad)
angle_max = 56.0 * np.pi / 180.0 # Maximum shooter angle (rad)

# Ball Constants
m = 0.215 # Mass of the ball (kg)
R = 0.0750 # Radius of the ball (m)
A = np.pi * R**2 # Cross-sectional area of the ball (m^2)
C_d = 0.5 # Drag coefficient (constant)
k = 0.3 # S -> C_l approximate ratio (constant)

# Global Constants
g = 9.81 # Gravitational acceleration (m/s^2)
p = 1.14 # Air density (kg/m^3)

# Generate set of values to 
speed = np.linspace(speed_min, speed_max, speed_density) # Speeds the bot is traveling (m/s)
dir = np.linspace(dir_min, dir_max, dir_density) # Directions the bot is traveling (rad)
rpm = np.linspace(rpm_min, rpm_max, rpm_density) # RPM's (rotations/min)
angle = np.linspace(angle_min, angle_max, angle_density) # Angle's (rad)

# -------------------- Variables --------------------



# -------------------- Data --------------------

speed_deg = 2 # Degree of speed polynomial
spin_deg = 2 # Degree of spin polynomial

speed_c = [] # Coefficients for speed polynomial (NEEDS TO BE INITIALIZED BEFORE DATATABLE)
spin_c = [] # Coefficients for spin polynomial (NEEDS TO BE INITIALIZED BEFORE DATATABLE)

test_rpm = np.array([2000, 3000, 4000, 5000, 6000]) # RPM's (rotations/min)
test_speed = np.array([3.0, 4.5, 7.2, 9.0, 9.0]) # Speed exiting shooter (m/s)
test_spin = np.array([-3.43, -4.29, -4.00, -5.00, -6.32]) # Spin exiting shooter (rotations/s)

num_tests = 1 # Number of tests
timestep = 0.033333 # Timestep between snapshots (s)
tests = np.array([[3000, 51 * np.pi / 180]]) # [Motor RPM (rotations/s), Hood Angle (rad)]
test_x = np.array([[0.00000, 0.11136, 0.22727, 0.34545, 0.46364, 0.59318, 0.67045, 0.77727, 0.89318, 1.00000, 1.11818, 1.22045, 1.32727, 1.43409]]) # X positions each frame (m)
test_y = np.array([[0.39000, 0.51727, 0.64455, 0.76273, 0.86500, 0.95818, 1.03318, 1.09227, 1.16500, 1.22409, 1.27182, 1.29682, 1.32182, 1.34455]]) # Y positions each frame (m)
t = np.arange(0, test_x.shape[1]) * timestep

# -------------------- Data --------------------



# -------------------- Setup --------------------

def rpm_to_params(rpm): # Convert RPM to speed and spin
    speed = 0
    spin = 0
    for i in range(speed_deg + 1):
        speed += speed_c[i] * rpm**(speed_deg - i)
    for i in range(spin_deg + 1):
        spin += spin_c[i] * rpm**(spin_deg - i)

    return speed, spin

def ode(t, y): # ODE's for integrator
    v = y[3:6]

    # Helper values
    speed = np.linalg.norm(v)
    S = y[6] * R / speed
    C_l = k * S # APPROXIMATION

    # Forces
    drag = -1 / 2 * p * C_d * A * speed * v
    # magnus = 1 / 2 * p * C_l * A * speed**2 * np.cross(np.array([0, 1, 0]), v / speed)
    magnus = 1 / 2 * p * C_l * A * speed * np.array([v[2], 0.0, -v[0]])
    gravity = np.array([0, 0, -m * g])

    # Differentials
    dydt = np.zeros(7)
    dydt[0:3] = v
    dydt[3:6] = (drag + magnus + gravity) / m
    dydt[6] = 0

    return dydt

def ode_c(t, y, C_d, k): # ODE's for integrator
    v = y[3:6]

    # Helper values
    speed = np.linalg.norm(v)
    if speed == 0:
        speed = 1e-8

    S = y[6] * R / speed
    C_l = k * S # APPROXIMATION

    # Forces
    drag = -1 / 2 * p * C_d * A * speed * v
    # magnus = 1 / 2 * p * C_l * A * speed**2 * np.cross(np.array([0, 1, 0]), v / speed)
    magnus = 1 / 2 * p * C_l * A * speed * np.array([v[2], 0.0, -v[0]])
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

def simulate(s, d, r, a): # Get the landing x for certain initial conditions
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
    C_d, k = x
    e = 0

    for test in tests:
        r, a = test
        speed, spin = rpm_to_params(r)

        # Set up integrator
        y0 = np.array([0.0, 0.0, shooter_height, speed * np.cos(a), 0.0, speed * np.sin(a), spin])
        sol = solve_ivp(ode_c, args=(C_d, k), t_span=(0.0, t[-1]), y0=y0, t_eval=t, rtol=1e-8, atol=1e-10)

        est_x = sol.y[0]
        est_y = sol.y[2]

        # Calculate error between estimation and data
        for i in range(num_tests):
            row_x = test_x[i]
            row_y = test_y[i]

            e += np.sum((row_x - est_x)**2)
            e += np.sum((row_y - est_y)**2)
    
    return e

def optimize_consts(): # Optimize constants using test data
    global C_d, k

    x0 = [C_d, k]
    result = minimize(error, x0=x0)
    C_d, k = result.x

# -------------------- Calculation --------------------



# -------------------- Lookup --------------------

def gen_lookup(): # Create the lookup table
    # Percentage visual
    percent_ratio = rpm_density * angle_density * speed_density * dir_density
    current_iter = 0
    percent_done = -1

    # Integrate
    datatable = []
    for i_1 in range(speed_density):
        for i_2 in range(dir_density):
            for i_3 in range(rpm_density):
                for i_4 in range(angle_density):
                    # Show perentage
                    current_iter += 1
                    percent = int(current_iter / percent_ratio * 100)
                    if percent != percent_done:
                        bar = int(current_iter / percent_ratio * 20)
                        print(f'\rIntegrating [{'-' * bar}{' ' * (20 - bar)}] %{percent}', end='')
                        percent_done = percent

                    # Populate table
                    s = speed[i_1]
                    d = dir[i_2]
                    r = rpm[i_3]
                    a = angle[i_4]

                    shot_dist, shot_dir = simulate(s, d, r, a)
                    if shot_dist:
                        x_vel = s * np.cos(d)
                        y_vel = s * np.sin(d)
                        datatable.append([shot_dist, shot_dir * 180 / np.pi, x_vel, y_vel, r, a * 180 / np.pi])

    print(f'\nForming lookup table with {len(datatable)} datapoints')

    # Sort table
    datatable = np.array(datatable)
    sorted = datatable[:, 0].argsort()

    # Write to CSV
    with open('shooter-lookup.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['landing distance', 'landing direction', 'bot x velocity', 'bot y velocity', 'rpm', 'angle'])
        for i in sorted:
            row = datatable[i, :]
            writer.writerow(row)

# -------------------- Lookup --------------------



# -------------------- Graph --------------------

def sim_path(s, d, r, a): # Plot the path of a projectile with certain starting conditions
    speed, spin = rpm_to_params(r)

    # Setup integrator
    y0 = np.array([0.0, 0.0, shooter_height, s * np.cos(d) + speed * np.cos(a), s * np.sin(d), speed * np.sin(a), spin])
    sol = solve_ivp(ode, t_span=(0, t_max), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10)

    # Plot data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0, :], sol.y[1, :], sol.y[2, :])
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

def plot_shot_data(): # Plot the collected speed and spin datapoints against the polynomial curves
    # Get data
    x = np.linspace(rpm_min, rpm_max, 100)
    y1, y2 = rpm_to_params(x)

    # Plot data
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.scatter(test_rpm, test_speed)
    plt.scatter(test_rpm, test_spin)

    plt.show()

def plot_test_shots():
    for i in range(num_tests):
        r, a = tests[i]
        speed, spin = rpm_to_params(r)

        # Setup integrator
        y0 = np.array([0.0, 0.0, shooter_height, speed * np.cos(a), 0.0, speed * np.sin(a), spin])
        sol = solve_ivp(ode, t_span=(0.0, t[-1]), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10, max_step=0.01)

        # Plot data
        plt.plot(sol.y[0, :], sol.y[2, :])
    
    for i in range(num_tests):
        plt.scatter(test_x[i], test_y[i])
    
    plt.show()

# -------------------- Graph --------------------



fit_shot_data()
# optimize_consts()
gen_lookup()