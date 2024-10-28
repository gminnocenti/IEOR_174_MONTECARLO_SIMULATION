import numpy as np
import pandas as pd

from functions_for_simulation import simulate_products_all_orders, rejection_sampling, generate_matrix_simulation

def ant_colony_vrp(distance_matrix, alpha, beta, num_ants, evaporation_rate, num_iterations):
    # Number of locations (including the starting depot)
    num_locations = len(distance_matrix)
    
    # Initialize pheromone matrix with a small constant value
    pheromone_matrix = np.full((num_locations, num_locations), 1.0)

    # Ant paths and distances
    best_route = None
    best_route_length = float('inf')

    # Main ACO loop
    for iteration in range(num_iterations):
        # Track all ant routes for this iteration
        all_routes = []
        all_distances = []

        for ant in range(num_ants):
            # Randomly select starting point (distribution center)
            current_location = 0  # Assuming 0 is the depot index
            visited = {current_location}
            route = [current_location]
            total_distance = 0

            # Construct the ant's route
            for step in range(num_locations - 1):
                # Calculate probabilities for the next location
                probabilities = []
                for next_location in range(num_locations):
                    if next_location not in visited:
                        pheromone_level = pheromone_matrix[current_location][next_location]
                        heuristic_value = 1 / (distance_matrix[current_location][next_location] + 1e-10)  # Avoid division by zero
                        probability = (pheromone_level ** alpha) * (heuristic_value ** beta)
                        probabilities.append((next_location, probability))
                
                # Normalize probabilities
                total_prob = sum(prob for _, prob in probabilities)
                probabilities = [(loc, prob / total_prob) for loc, prob in probabilities]
                
                # Choose the next location based on the probabilities
                next_location = np.random.choice([loc for loc, _ in probabilities],
                                                 p=[prob for _, prob in probabilities])

                # Update route and distances
                route.append(next_location)
                total_distance += distance_matrix[current_location][next_location]
                visited.add(next_location)
                current_location = next_location

            # Return to the depot to complete the route
            route.append(0)
            total_distance += distance_matrix[current_location][0]

            # Save the route and distance
            all_routes.append(route)
            all_distances.append(total_distance)

            # Update the best solution if this route is shorter
            if total_distance < best_route_length:
                best_route_length = total_distance
                best_route = route

        # Pheromone evaporation
        pheromone_matrix *= (1 - evaporation_rate)

        # Update pheromone matrix based on the routes found
        for route, distance in zip(all_routes, all_distances):
            pheromone_to_add = 1.0 / distance
            for i in range(len(route) - 1):
                from_loc = route[i]
                to_loc = route[i + 1]
                pheromone_matrix[from_loc][to_loc] += pheromone_to_add
                pheromone_matrix[to_loc][from_loc] += pheromone_to_add  # For symmetric distances

        # Print progress
        print(f"Iteration {iteration + 1}/{num_iterations}, Best Route Length: {best_route_length}")

    return best_route, best_route_length

# example
np.random.seed(42)
df_number_of_orders_to_deliver = pd.read_csv("SIM_density_number_of_orders_delivered_in_a_day.csv")
n_orders = rejection_sampling(df_number_of_orders_to_deliver["Number_of_orders_delivered"], 1)

distance_matrix_dc1 = np.loadtxt("SIM_distribution_center_1_distance_matrix.csv", delimiter=",")
distance_matrix_dc2 = np.loadtxt("SIM_distribution_center_2_distance_matrix.csv", delimiter=",")
sampled_distance_matrix_dc1, sampled_distance_matrix_dc2 = generate_matrix_simulation(
    distance_matrix_dc1, distance_matrix_dc2, n_orders[0]
)

best_route, best_length = ant_colony_vrp(sampled_distance_matrix_dc1, alpha=1, beta=1, num_ants=10, evaporation_rate=0.5, num_iterations=100)
print("Best Route:", best_route)
print("Best Route Length:", best_length)
