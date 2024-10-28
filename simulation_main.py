import pandas as pd
import numpy as np
import time
from functions_for_simulation import simulate_products_all_orders, rejection_sampling, generate_matrix_simulation

# Load Datasets
df_number_of_orders_to_deliver = pd.read_csv("SIM_density_number_of_orders_delivered_in_a_day.csv")
df_number_of_products_in_one_order = pd.read_csv("SIM_number_of_products_in_a_order.csv")
df_most_ordered_products = pd.read_csv("SIM_most_ordered_products.csv")
distance_matrix_distribution_center1 = np.loadtxt("SIM_distribution_center_1_distance_matrix.csv", delimiter=",")
distance_matrix_distribution_center2 = np.loadtxt("SIM_distribution_center_2_distance_matrix.csv", delimiter=",")

def simulate_vrp(iterations: int):
    """
    Simulates the whole VRP process.

    -> iterations: number of times the simulation will be made.
    """
    average_n_orders = []
    for _ in range(iterations):
        # Measure time for number of orders simulation
        start_time = time.time()
        n_orders = rejection_sampling(df_number_of_orders_to_deliver["Number_of_orders_delivered"], 1)  # approx 38
        print(f"N_orders calculated in {time.time() - start_time:.4f} seconds")

        # Measure time for size of orders simulation
        start_time = time.time()
        size_of_orders = rejection_sampling(df_number_of_products_in_one_order['Order_size_count'], int(n_orders[0]))
        print(f"Size of orders calculated in {time.time() - start_time:.4f} seconds")

        # Measure time for product IDs simulation
        start_time = time.time()
        product_id_for_each_order = simulate_products_all_orders(df_most_ordered_products, size_of_orders)
        print(f"Product id's for each order calculated in {time.time() - start_time:.4f} seconds")

        # Measure time for distance matrix simulation
        start_time = time.time()
        distance_matrix_simulation_DC_1, distance_matrix_simulation_DC_2 = generate_matrix_simulation(
            distance_matrix_distribution_center1, distance_matrix_distribution_center2, n_orders[0]
        )
        print(f"Distance matrix simulated in {time.time() - start_time:.4f} seconds")

        average_n_orders.append(n_orders)
        print("-" * 30)

    average_n_orders = np.array(average_n_orders)
    print(f"Average # orders in {iterations} days: {average_n_orders.mean()}")

simulate_vrp(1000)
