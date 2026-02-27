import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
# 44 -> 46 from horizontal - 2000/3000/4000/5000/6000 RPM
# 34 -> 56 from horizontal - 2000/3000/4000/5000/6000 RPM
# Sanity check: 39 -> 51 from horizontal - 3000 RPM



# -------------------- Variables --------------------

# Ball Constants
m = 0.215 # Mass of the ball (kg)
R = 0.075 # Radius of the ball (m)
A = np.pi * R**2 # Cross-sectional area of the ball (m^2)
C_d = 0.5 # Drag coefficient (constant)
k = 0.3 # S -> C_l approximate ratio (constant)

# Global Constants
g = 9.81 # Gravitational acceleration (m/s^2)
p = 1.14 # Air density (kg/m^3)

# Simulation Variables
shooter_height = 0.39 # Height of the shooter (m)
target_height = 1.83 + R # Height of the target (m)
angle_range = np.pi / 4.0 # Range of acceptable angles (rad)
t_max = 2.0 # Max time to integrate to (s)
velocity_density = 37 # Number of values used when simulating speeds
rpm_density = 121 # Number of values used when simulating rpms
angle_density = 21 # Number of values used when simulating hood angles

# Shooter Constraints
speed_max = 4.5 # Maximum bot speed (m/s)
rpm_min = 3000.0 # Minimum rpm (2pi*rad/min)
rpm_max = 6000.0 # Maximum rpm (2pi*rad/min)
angle_min = 46.0 # Minimum shooter angle (rad)
angle_max = 56.0 # Maximum shooter angle (rad)

# Generate set of values to simulate
velocity_x = np.linspace(-speed_max, speed_max, velocity_density) # Velocity of the bot to or from the target (m/s)
velocity_y = np.linspace(-speed_max, speed_max, velocity_density) # Velocity of the bot to the left or right of the target (m/s)
rpm = np.linspace(rpm_min, rpm_max, rpm_density) # RPM's (rotations/min)
angle_deg = np.linspace(angle_min, angle_max, angle_density) # Angle's (deg)
angle_rad = angle_deg * 180 / np.pi # Angle's (rad)

# -------------------- Variables --------------------



# -------------------- Data --------------------

speed_deg = 2 # Degree of speed polynomial
spin_deg = 2 # Degree of spin polynomial

speed_c = [] # Coefficients for speed polynomial (NEEDS TO BE INITIALIZED BEFORE DATATABLE)
spin_c = [] # Coefficients for spin polynomial (NEEDS TO BE INITIALIZED BEFORE DATATABLE)

test_rpm = np.array([3000.0, 4000.0, 5000.0, 6000.0, 3000.0, 4000.0, 5000.0, 6000.0]) # RPM's (rotations/min)
test_speed = np.array([5.561746945475823, 6.9015543559842465, 7.827967590669779, 8.083195075204472, 5.131013479562509, 7.099633453821274, 7.927967590669779, 8.36101844274653]) # Speed exiting shooter (m/s)
test_spin = np.array([-19.84163781214606, -26.92793703076965, -31.415926535897928, -41.8879020478639, -22.175948142986773, -28.999316802367318, -31.415926535897928, -34.27191985734319]) # Spin exiting shooter (rotations/s)

num_tests = 2 # Number of tests
timestep = 0.03333333333333333 # Timestep between snapshots (s)
tests = np.array([[3000.0, 46.0 * np.pi / 180.0], [3000.0, 56.0 * np.pi / 180.0]]) # [Motor RPM (rotations/s), Hood Angle (rad)]
test_x = np.array([[0.0, 0.13738317757009344, 0.2719626168224299, 0.40934579439252333, 0.5397196261682242, 0.667289719626168, 0.8018691588785045, 0.916822429906542, 1.0499999999999998, 1.1831775700934577, 1.3079439252336447, 1.4467289719626166, 1.5799065420560745, 1.7074766355140185, 1.8476635514018689, 1.986448598130841, 2.1294392523364483],
                   [0.0, 0.10654205607476636, 0.2116822429906542, 0.31542056074766356, 0.42757009345794394, 0.46542056074766347, 0.5677570093457943, 0.6728971962616822, 0.7738317757009344, 0.8747663551401869, 0.9742990654205607, 1.0710280373831775, 1.1719626168224297, 1.2855140186915888, 1.3962616822429905, 1.5042056074766355, 1.6191588785046727]]) # X positions each frame (m)
test_y = np.array([[0.39, 0.5189719626168224, 0.6409345794392522, 0.7488785046728972, 0.8442056074766354, 0.9325233644859813, 1.004018691588785, 1.068504672897196, 1.128785046728972, 1.1806542056074765, 1.2157009345794392, 1.2423364485981307, 1.2647663551401869, 1.2745794392523364, 1.270373831775701, 1.2577570093457942, 1.2339252336448596],
                   [0.39, 0.5245794392523365, 0.6605607476635514, 0.7867289719626168, 0.8960747663551402, 0.9395327102803738, 1.0320560747663552, 1.1189719626168224, 1.1932710280373833, 1.256355140186916, 1.31803738317757, 1.3600934579439252, 1.403551401869159, 1.4273831775700936, 1.454018691588785, 1.4624299065420558, 1.4610280373831777]]) # Y positions each frame (m)
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

def simulate(vx, vy, r, a): # Get the landing x for certain initial conditions
    speed, spin = rpm_to_params(r)

    # Set up integrator
    y0 = np.array([0.0, 0.0, shooter_height, vx + speed * np.cos(a), vy, speed * np.sin(a), spin])
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
    percent_ratio = velocity_density**2 * rpm_density * angle_density
    current_iter = 0
    percent_done = -1

    # Integrate
    datatable = []
    for i_1 in range(velocity_density):
        vx = velocity_x[i_1]
        for i_2 in range(velocity_density):
            vy = velocity_y[i_2]

            if vx**2 + vy**2 > speed_max**2:
                current_iter += rpm_density * angle_density
                continue

            for i_3 in range(rpm_density):
                r = rpm[i_3]
                for i_4 in range(angle_density):
                    a_rad = angle_rad[i_4]
                    a_deg = angle_deg[i_4]

                    # Show perentage
                    current_iter += 1
                    percent = int(current_iter / percent_ratio * 100)
                    if percent != percent_done:
                        bar = int(current_iter / percent_ratio * 20)
                        print(f'\rIntegrating [{'-' * bar}{' ' * (20 - bar)}] %{percent}', end='')
                        percent_done = percent

                    shot_dist, shot_dir = simulate(vx, vy, r, a_rad)
                    if shot_dist:
                        datatable.append([shot_dist, shot_dir * 180.0 / np.pi, vx, vy, r, a_deg - angle_min])

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

def sim_path(vx, vy, r, a): # Plot the path of a projectile with certain starting conditions
    speed, spin = rpm_to_params(r)

    # Setup integrator
    y0 = np.array([0.0, 0.0, shooter_height, vx + speed * np.cos(a), vy, speed * np.sin(a), spin])
    sol = solve_ivp(ode, t_span=(0, t_max), y0=y0, rtol=1e-8, atol=1e-10)

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
# optimize_consts() # Innacurate due to plots not accounting for perspective
gen_lookup()