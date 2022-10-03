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

# Load spice kernels
spice.load_standard_kernels()

# Set simulation start epochs, 1 JANUARY 2023 START
start_epoch = 725803200.000000
tof = 1.5*constants.JULIAN_YEAR

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

initial_state = bodies.get_body('Earth').state_in_base_frame_from_ephemeris(start_epoch)
final_state = bodies.get_body('Mars').state_in_base_frame_from_ephemeris(start_epoch+tof)

final_time = start_epoch+tof
frequency = 2.0 * np.pi / tof
scale_factor = 1.0 / tof
number_of_revolutions = int(0)

AU =  149597870700.0
departure_semi_major_axis = np.inf
departure_eccentricity = 0
arrival_semi_major_axis = np.inf
arrival_eccentricity = 0


Results = []

for i in range(0,500):
    print(i)
    #                    C4        C5         C9       C10         C14      C15

    #free_coefficients = [uniform(-3900, 2300), uniform(-3800, 8600), uniform(-6300, 4500),
    #                 uniform(-7700, 4200), uniform(-1200, 2100), uniform(-4800, -50)]
    #free_coefficients = [uniform(-5800, 8500), uniform(-8200, 14000), uniform(-12000, 6000),
    #                     uniform(-17000, 16000), uniform(-6200, 4500), uniform(-5600, -5900)]

    free_coefficients = [uniform(-58000, 12000), uniform(-2500, 17000), uniform(-43000, -1300), uniform(-86000, 15000),
                         uniform(-17000, 11000), uniform(-10000, 9300)]

    radial_free_coefficients = free_coefficients[0:2]
    normal_free_coefficients = free_coefficients[2:4]
    axial_free_coefficients = free_coefficients[4:6]

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

    transfer_leg_settings = []
    transfer_leg_settings.append( transfer_trajectory.hodographic_shaping_leg(
         radial_velocity_shaping_functions,
         normal_velocity_shaping_functions,
         axial_velocity_shaping_functions))

    leg_type = []
    leg_type.append(transfer_trajectory.TransferLegTypes.unpowered_unperturbed_leg_type)

    transfer_node_settings = []
    transfer_node_settings.append( transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity) )
    transfer_node_settings.append( transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity) )
    transfer_body_order = ['Earth','Mars']

    # transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_hodographic_shaping_legs(
    #             body_order=transfer_body_order,
    #             radial_velocity_function_components_per_leg=[radial_velocity_shaping_functions],
    #             normal_velocity_function_components_per_leg=[normal_velocity_shaping_functions],
    #             axial_velocity_function_components_per_leg=[axial_velocity_shaping_functions],
    #             departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    #             arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity) )



    transfer = transfer_trajectory.create_transfer_trajectory(bodies,
                                                              transfer_leg_settings,
                                                              transfer_node_settings,
                                                              transfer_body_order,
                                                              central_body = 'Sun')

    #transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

    node_times = list()
    node_times.append(start_epoch)
    node_times.append(start_epoch+tof)

    node_parameters = list()
    node_parameters.append(np.zeros(3))
    node_parameters.append(np.zeros(3))


    leg_parameters = [number_of_revolutions]
    leg_parameters.extend(free_coefficients)

    transfer.evaluate(node_times, [leg_parameters], node_parameters)

    # print('Delta V [m/s]: ', transfer.delta_v)
    # print('Time of flight [day]: ', transfer.time_of_flight / constants.JULIAN_DAY)
    # print('Delta V per leg [m/s] : ', transfer.delta_v_per_leg)
    # print('Delta V per node [m/s] : ', transfer.delta_v_per_node)

    state_history = transfer.states_along_trajectory(1000)
    node_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
    state_history = result2array(state_history)
    #print(state_history
    #      )
    leg_parameters.append(transfer.delta_v)
    Results.append(leg_parameters)


#print(Results)


ResultsPD = pd.DataFrame(Results)

ResultsPD.to_csv('D:\Documenten\Studiedocumenten\Master\Thesis\Python\stnewver.csv')

    #save2txt(state_history,
    #         'OnlyRecommend.dat',
    #        'D:\Documenten\Studiedocumenten\Master\Thesis\Python'
    #        )

