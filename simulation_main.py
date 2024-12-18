import pandas as pd
import numpy as np
import time
from functions_for_simulation import simulate_products_all_orders, rejection_sampling, generate_matrix_simulation, calculate_volume_for_each_order,loading_truck_algorithm, calculate_distance_matrix_for_each_truck,calculate_vrp

# Load Datasets
df_number_of_orders_to_deliver = pd.read_csv("SIM_density_number_of_orders_delivered_in_a_day.csv")
df_number_of_products_in_one_order = pd.read_csv("SIM_number_of_products_in_a_order.csv")
df_most_ordered_products = pd.read_csv("SIM_most_ordered_products.csv")
distance_matrix_distribution_center1 = np.loadtxt("SIM_distribution_center_1_distance_matrix.csv", delimiter=",")
distance_matrix_distribution_center2 = np.loadtxt("SIM_distribution_center_2_distance_matrix.csv", delimiter=",")
TRUCK_CAPACITY = 1821760
N_TRUCKS=3
SPEED_KMH=30 #km\hr
WORK_HOURS_PER_DAY=8 #hr
# DISTANCE MATRIX IS IN KM
def simulate_vrp(iterations: int):
    """
    Simulates the whole VRP process.

    -> iterations: number of times the simulation will be made.
    """
    #store results of each iteration
    final_results_DC_1=[]
    final_results_DC_2=[]
    for _ in range(iterations):
        n_orders = rejection_sampling(df_number_of_orders_to_deliver["Number_of_orders_delivered"], 1)  # approx 38

        
        size_of_orders = rejection_sampling(df_number_of_products_in_one_order['Order_size_count'], int(n_orders[0]))

        # Measure time for product IDs simulation
        product_id_for_each_order = simulate_products_all_orders(df_most_ordered_products, size_of_orders)

        # Measure time for distance matrix simulation
        distance_matrix_simulation_DC_1, distance_matrix_simulation_DC_2 = generate_matrix_simulation(
            distance_matrix_distribution_center1, distance_matrix_distribution_center2, n_orders[0]
        )

        # calculate volume for each order
        volume_for_each_order=calculate_volume_for_each_order(product_id_for_each_order)

        # find the number of necessary trucks to full fill the order and how many orders each truck will deliver
        number_of_orders_each_truck_will_deliver=loading_truck_algorithm(TRUCK_CAPACITY,volume_for_each_order)

        # calculate distance matrix for each truck and distribution center
        list_distance_matrix_DC_1=calculate_distance_matrix_for_each_truck(number_of_orders_each_truck_will_deliver,distance_matrix_simulation_DC_1)
        list_distance_matrix_DC_2=calculate_distance_matrix_for_each_truck(number_of_orders_each_truck_will_deliver,distance_matrix_simulation_DC_2)
        
        #calculate VRP for each DC
        result_DC_1=calculate_vrp(list_distance_matrix_DC_1, N_TRUCKS,SPEED_KMH,WORK_HOURS_PER_DAY)
        result_DC_2=calculate_vrp(list_distance_matrix_DC_2, N_TRUCKS,SPEED_KMH,WORK_HOURS_PER_DAY)
        final_results_DC_1.append(result_DC_1)
        final_results_DC_2.append(result_DC_2)
    return final_results_DC_1,final_results_DC_2

         
final_results_DC_1,final_results_DC_2=simulate_vrp(300)


columns = [
    "Number of Completed Trips",
    "Number of Uncompleted Trips",
    "Number of Completed Orders",
    "Number of Uncompleted Orders",
    "Total Distance Completed Orders (km)",
    "Total Distance Uncompleted Orders (km)",
    "Total Time Completed Orders (hrs)",
    "Total Time Uncompleted Orders (hrs)",
    "Average Distance for Successful Orders (km)",
    "Average Time for Successful Orders (hrs)"
]


df_results_DC_1 = pd.DataFrame(final_results_DC_1, columns=columns)

df_results_DC_2 = pd.DataFrame(final_results_DC_2, columns=columns)

#df_results_DC_1.to_csv("results_DC1_3_trucks.csv")
#df_results_DC_2.to_csv("results_DC2_3_trucks.csv")