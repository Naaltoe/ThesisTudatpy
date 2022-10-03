# Load standard modules
import numpy as np
# make plots interactive

from matplotlib import pyplot as plt

# Load tudatpy modules
import pandas as pd
from tudatpy.kernel.interface import spice
from random import randrange, uniform
from tudatpy.io import save2txt
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load spice kernels
spice.load_standard_kernels()

# Set simulation start epochs, 1 JANUARY 2023 START
start_epoch = 725803200.000000
tof = 1.5*constants.JULIAN_YEAR
final_time = start_epoch+tof

# Define settings for celestial bodies
bodies_to_create = ['Earth',
                    'Mars',
                    'Sun']

# Define coordinate system
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'
# Create body settings
body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                            global_frame_origin,
                                                            global_frame_orientation)
# Create bodies
bodies = environment_setup.create_simplified_system_of_bodies()
#bodies = environment_setup.create_system_of_bodies(body_settings)

Central_body = [
    "Sun"
]


#Set parameters required for defining velocity functions
frequency = 2.0 * np.pi / tof
scale_factor = 1.0 / tof
number_of_revolutions = int(0)

# Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
radial_velocity_shaping_functions = shape_based_thrust.recommended_radial_hodograph_functions(tof)
# Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)

radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
    exponent=1.0,
    frequency= 0.5*frequency,
    scale_factor=scale_factor))
radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
    exponent=1.0,
    frequency=0.5 * frequency,
    scale_factor=scale_factor))

# Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
normal_velocity_shaping_functions = shape_based_thrust.recommended_normal_hodograph_functions(tof)
# Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
    exponent=1.0,
    frequency=0.5 * frequency,
    scale_factor=scale_factor))
normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
    exponent=1.0,
    frequency=0.5 * frequency,
    scale_factor=scale_factor))


axial_velocity_shaping_functions = shape_based_thrust.recommended_axial_hodograph_functions(
    tof,
    number_of_revolutions)
# Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
    exponent=4.0,
    frequency=(number_of_revolutions + 0.5) * frequency,
    scale_factor=scale_factor**4.0))
axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
    exponent=4.0,
    frequency=(number_of_revolutions + 0.5) * frequency,
    scale_factor=scale_factor**4.0))


# Set transfer leg settings using the above created shaping functions
transfer_leg_settings = []
transfer_leg_settings.append( transfer_trajectory.hodographic_shaping_leg(
     radial_velocity_shaping_functions,
     normal_velocity_shaping_functions,
     axial_velocity_shaping_functions))

# Define leg type
leg_type = []
leg_type.append(transfer_trajectory.TransferLegTypes.hodographic_low_thrust_leg)


# Define departure and arrival orbit parameters
AU =  149597870700.0
departure_semi_major_axis = np.inf
departure_eccentricity = 0
arrival_semi_major_axis = np.inf
arrival_eccentricity = 0

# Define node settings and transfer order
transfer_node_settings = []
transfer_node_settings.append( transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity) )
transfer_node_settings.append( transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity) )
transfer_body_order = ['Earth','Mars']

# Create the transfer object
transfer = transfer_trajectory.create_transfer_trajectory(bodies,
                                                          transfer_leg_settings,
                                                          transfer_node_settings,
                                                          transfer_body_order,
                                                          central_body = 'Sun')

transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

# Set node times and parameters
node_times = list()
node_times.append(start_epoch)
node_times.append(start_epoch+tof)

# Parameter 2: Node 0 Outgoing excess velocity magnitude
# Parameter 3: Node 0 Outgoing excess velocity in-plane angle
# Parameter 4: Node 0 Outgoing excess velocity out-of-plane angle
# Parameter 5: Node 1 Incoming excess velocity magnitude
# Parameter 6: Node 1 Incoming excess velocity in-plane angle
# Parameter 7: Node 1 Incoming excess velocity out-of-plane angle
node_parameters = list()
node_parameters.append(np.zeros(3))
node_parameters.append(np.zeros(3))

# Set values for free coefficients
# Best coefficients found for N = 0
free_coefficients = [ -38852.8007796046,
                      12759.6520758622 ,
                      -31425.1033837461,
                      -54221.2080529588,
                      -9658.99274172873,
                      6519.19424919116]

# Set values for free coefficients
# Best coefficients found for N = 2
# free_coefficients = [ 8403.00136927371,
#                       -8031.5331631239,
#                       -5215.6553329413,
#                       -16685.2948543503,
#                       -2263.09603961912,
#                       -5813.16054843846]
radial_free_coefficients = free_coefficients[0:2]
normal_free_coefficients = free_coefficients[2:4]
axial_free_coefficients = free_coefficients[4:6]

# Create leg parameters and evaluate the transfer.
leg_parameters = [number_of_revolutions]
leg_parameters.extend(free_coefficients)

transfer.evaluate(node_times, [leg_parameters], node_parameters)


# Print and save results
print('Delta V [m/s]: ', transfer.delta_v)
print('Time of flight [day]: ', transfer.time_of_flight / constants.JULIAN_DAY)
print('Delta V per leg [m/s] : ', transfer.delta_v_per_leg)
print('Delta V per node [m/s] : ', transfer.delta_v_per_node)

state_history = transfer.states_along_trajectory(1000)
node_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)




# This is used to create the orbits of Earth and Mars in order to visualize the accuracy of the trajectory.
steps = 400
stepsize = tof/steps
Earthorbit= []
Marsorbit = []
for i in range(steps+1):
    Earthorbit.append(bodies.get_body('Earth').state_in_base_frame_from_ephemeris(start_epoch+i*stepsize))
    Marsorbit.append(bodies.get_body('Mars').state_in_base_frame_from_ephemeris(start_epoch+i*stepsize*2))



# Plot the transfer trajectory
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(state_history[:, 1] / AU, state_history[:, 2]/AU, state_history[:,3] / AU)
# Plot location of the Sun
ax.scatter3D([0],[0],[0],color='orange')

# Plot orbits of Earth and MArs
for i in range(len(Earthorbit)):
    ax.scatter3D(Earthorbit[i][0] / AU, Earthorbit[i][1] / AU, Earthorbit[i][2] / AU, color = 'green', s=0.5)
for i in range(len(Marsorbit)):
    ax.scatter3D(Marsorbit[i][0] / AU, Marsorbit[i][1] / AU, Marsorbit[i][2] / AU, color = 'red', s=0.5)

ax.set_xlabel('x wrt Sun [AU]')
ax.set_ylabel('y wrt Sun [AU]')
ax.set_zlabel('z wrt Sun [AU]')
ax.legend()
plt.tight_layout()
plt.show()



