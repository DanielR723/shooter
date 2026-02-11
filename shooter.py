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

speed_deg = 3 # Degree of speed polynomial
spin_deg = 3 # Degree of spin polynomial

speed_c = [] # Coefficients for speed polynomial (NEEDS TO BE INITIALIZED BEFORE DATATABLE)
spin_c = [] # Coefficients for spin polynomial (NEEDS TO BE INITIALIZED BEFORE DATATABLE)

test_rpm = np.array([100, 120, 140, 160, 180, 200]) # RPM's (2pi*rad/min)
test_speed = np.array([10, 20, 40, 65, 70, 100]) # Speed exiting shooter (m/s)
test_spin = np.array([100, 140, 145, 300, 325, 350]) # Spin exiting shooter (2pi*rad/s)

# -------------------- Data --------------------



# -------------------- Setup --------------------

def rpm_to_params(rpm): # Convert RPM to speed and spin
    speed = np.zeros(rpm.shape)
    spin = np.zeros(rpm.shape)
    for i in range(speed_deg + 1):
        speed += speed_c[i] * rpm**(speed_deg - i)
    for i in range(spin_deg + 1):
        spin += spin_c[i] * rpm**(speed_deg - i)

    return speed, spin

def ode(t, y): # ODE's for integrator
    v = y[3:6]

    # Helper values
    speed = np.linalg.norm(v)
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

def simulate(rpm, angle): # Get the landing x of a certain rpm and angle
    speed, spin = rpm_to_params(rpm)

    # Set up integrator
    y0 = np.array([0.0, 0.0, shooter_height, speed * np.cos(angle), 0.0, speed * np.sin(angle), spin])
    sol = solve_ivp(ode, t_span=(0, t_max), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10)

    # Get return value
    vertical = abs(np.arctan2(sol.y[3, -1], -sol.y[5, -1])) < angle_range
    if sol.t_events[0].size == 1 and vertical:
        return sol.y[0, -1]
    else:
        return None

def fit_data(): # Fit a polynomial to data collected for speed and spin
    global speed_c, spin_c

    speed_c = np.polyfit(test_rpm, test_speed, speed_deg)
    spin_c = np.polyfit(test_rpm, test_spin, spin_deg)

def plot_data(): # Plot the collected speed and spin datapoints against the polynomial curves
    x = np.linspace(rpm_min, rpm_max, rpm_density)
    y1, y2 = rpm_to_params(x)

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.scatter(test_rpm, test_speed)
    plt.scatter(test_rpm, test_spin)

    plt.show()

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



# -------------------- Graph --------------------

def graph(rpm, angle): # Graph a shot with set rpm and angle
    speed, spin = rpm_to_params(rpm)

    y0 = np.array([0.0, 0.0, shooter_height, speed * np.cos(angle), 0.0, speed * np.sin(angle), spin])
    sol = solve_ivp(ode, t_span=(0, t_max), y0=y0, events=hit_event, rtol=1e-8, atol=1e-10)

    plt.plot(sol.y[0, :], sol.y[2, :])
    plt.axis('equal')
    plt.show()

# -------------------- Graph --------------------



fit_data()
plot_data()