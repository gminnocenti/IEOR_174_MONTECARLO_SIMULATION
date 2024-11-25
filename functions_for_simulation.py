
from scipy.stats import gaussian_kde
import numpy as np
import math
import time
import json
class PseudoRandomNumberGenerator:
    def __init__(self):
        # Parameters for LCG (values from Numerical Recipes)
        self.a = 1664525  # Multiplier
        self.c = 1013904223  # Increment
        self.m = 2**32  # Modulus
        # Initial seed is generated from the current time
        self.state = int(time.time() * 1000) % self.m

    def random(self):
        # Linear Congruential Generator formula
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m


def simulate_id_to_fullfill_order(df_most_ordered_products,n_products):
    """Simulate the products ids that will make up an order
    """
    product_ids=[]
    prng = PseudoRandomNumberGenerator()
    for i in range(n_products):
        given_probability=np.random.rand()
# Use logical conditions to find the row where the probability is between 'Lim_inf' and 'Lim_sup'
        matching_row = df_most_ordered_products[(df_most_ordered_products['Lim_inf'] <= given_probability) & 
                                                    (df_most_ordered_products['Lim_sup'] > given_probability)]

        # Extract the value of 'product_id' for that row
        product_id_value = matching_row['product_id'].values[0]  
        product_ids.append(product_id_value)
    return product_ids

def simulate_products_all_orders(df_most_ordered_products,size_of_orders):
    """simulate the product ids of every order for the day
    """
    all_orders=[]
    for i in range(len(size_of_orders)):
        n_products=size_of_orders[i]
        all_orders.append(simulate_id_to_fullfill_order(df_most_ordered_products,n_products))
    
    return all_orders

class SelfRV:
    def __init__(self, kde):
        self.kde = kde  # Pass the KDE object to the class
        self.xmin = min(kde.dataset[0])  # Set xmin based on the KDE's dataset
        self.xmax = max(kde.dataset[0])  # Set xmax based on the KDE's dataset

        # Generate an array of values to evaluate the KDE and find the maximum
        x_list = np.linspace(self.xmin, self.xmax, 1000)
        self.max_pdf = max(self.kde(x_list)) * 1.1  # Adjust scaling if necessary

    # Use the KDE for the pdf function
    def pdf(self, x):
        if x < self.xmin or x > self.xmax:
            return 0
        return self.kde(x)[0]  # Return the KDE value at x
    
    def sample_one(self):
        # Perform rejection sampling
        x_0 = np.random.uniform(self.xmin, self.xmax)
        keep_probability = self.pdf(x_0) / self.max_pdf
        whether_keep = np.random.binomial(n=1, p=keep_probability)
        while not whether_keep:
            x_0 = np.random.uniform(self.xmin, self.xmax)
            keep_probability = self.pdf(x_0) / self.max_pdf
            whether_keep = np.random.binomial(n=1, p=keep_probability)
        return x_0

    def sample(self, size):
        return np.array([self.sample_one() for i in range(size)])

def generate_matrix_simulation(distance_matrix_distribution_center1,distance_matrix_distribution_center2,n_cities):
    """generate a distance matrix for the number of orders to fullfill in a day
    """
    # Select n-1 unique random cities excluding the first city (city at index 0)
    random_cities = np.random.choice(np.arange(1, distance_matrix_distribution_center1.shape[0]), size=n_cities-1, replace=False)

    # Add the first city (index 0) to the selected cities
    selected_cities = np.concatenate(([0], random_cities))

    # Create a new distance matrix with the selected cities
    selected_cities_distribution_center_1 = distance_matrix_distribution_center1[np.ix_(selected_cities, selected_cities)]
    selected_cities_distribution_center_2 = distance_matrix_distribution_center2[np.ix_(selected_cities, selected_cities)]

    return selected_cities_distribution_center_1,selected_cities_distribution_center_2

def rejection_sampling(data,n_samples):
    """rejection sampling
    """
    kde = gaussian_kde(data)
    sampler = SelfRV(kde)
    samples = sampler.sample(n_samples)
    return np.int16(samples) 

def simulate_number_of_products_in_a_order(x_order_sizes,weights,n_samples):
    # Create a weighted KDE based on order sizes
    kde = gaussian_kde(x_order_sizes, weights=weights)
    self_rv = SelfRV(kde)

# Generate a sample of size 1000
    samples = self_rv.sample(n_samples)
    return np.int16(samples) 

def simulate_number_of_orders_in_a_day(x_n_orders,weights,n_samples):
    kde = gaussian_kde(x_n_orders, weights=weights)
    self_rv = SelfRV(kde)

    # Generate a sample of size 1000
    samples =  self_rv.sample(n_samples)
    return np.int16(samples) 



def calculate_volume_for_each_order(product_id_for_each_order, json_file_path='product_dimensions.json'):
    """
    Calculate the volume for each product in an order using a precomputed JSON dictionary
    """
    # Load the precomputed dictionary from JSON
    with open(json_file_path, 'r') as json_file:
        product_dimensions = json.load(json_file)
    
    # Calculate volumes
    volume_for_each_order = [
        [product_dimensions[product_id]['length'] *
         product_dimensions[product_id]['height'] *
         product_dimensions[product_id]['width']
         for product_id in order]
        for order in product_id_for_each_order
    ]
    
    return volume_for_each_order

def loading_truck_algorithm(TRUCK_CAPACITY,volume_for_each_order): 
    """
    This functions loads the orders into a truck. First it finds the number of trucks necessary to deliver the orders of the day based on the TRUCK_CAPACITY.
    It returns the a list containing the number of orders each truck will deliver.
    """
    trucks = []  # List to store loaded trucks
    current_truck = []  # Current truck's orders
    current_capacity = TRUCK_CAPACITY  # Remaining capacity in the current truck

    for i in range(len(volume_for_each_order)):
        order=volume_for_each_order[i]
        total_order_volume = sum(order)
        
        # Check if the order fits in the current truck
        if total_order_volume <= current_capacity:
            # Add order to the current truck
            current_truck.append(i+1)
            current_capacity -= total_order_volume
        else:
            # If it doesn't fit, start a new truck
            trucks.append(current_truck)
            current_truck = [i]  # Start the new truck with this order
            current_capacity = TRUCK_CAPACITY - total_order_volume

    # Append the last truck to the list
    if current_truck:
        trucks.append(current_truck)
    # this list contains the number of orders each truck will deliver
    number_of_orders_each_truck_will_deliver = [len(trucks[0]) - 1 if i == 0 else len(sublist) for i, sublist in enumerate(trucks)]
    return number_of_orders_each_truck_will_deliver

def calculate_distance_matrix_for_each_truck(number_of_orders_each_truck_will_deliver,distance_matrix_simulation_DC):
    """
    This function creates a distance matrix for each truck based on the numbers of orders each truck will deliver given
    by the loading algorithm.
    """
    start_idx = 1  # Start after the distribution center (index 0)
    truck_matrices = []

    # Generate distance matrices for each truck
    for num_deliveries in number_of_orders_each_truck_will_deliver:
        truck_indices = [0] + list(range(start_idx, start_idx + num_deliveries))
        truck_matrix = distance_matrix_simulation_DC[np.ix_(truck_indices, truck_indices)]
        truck_matrices.append(truck_matrix)
        start_idx += num_deliveries
    
    return truck_matrices
def ant_colony_vrp(distance_matrix, alpha, beta, num_ants, evaporation_rate, num_iterations):
    """
    This is the vrp algorithm we used for the simulation
    """
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
            pheromone_to_add = 1.0 / (distance + 1e-10)  
            for i in range(len(route) - 1):
                from_loc = route[i]
                to_loc = route[i + 1]
                pheromone_matrix[from_loc][to_loc] += pheromone_to_add
                pheromone_matrix[to_loc][from_loc] += pheromone_to_add  # For symmetric distances

        # Print progress

    return best_route_length

def calculate_time(distance_km,SPEED_KMH):
    return distance_km / SPEED_KMH

# Simulation function
def calculate_vrp(distance_matrices, N_TRUCKS,SPEED_KMH,WORK_HOURS_PER_DAY):
    """
    This function calculates the vrp algorithm for each distance matrix for each truck of the specific orders to deliver in a day.
    """
    total_distance = 0
    truck_availability = [0] * N_TRUCKS  # Initialize truck availability with 0 hours used for all trucks
    number_completed_trips=0
    number_uncompleted_trips=0
    number_of_completed_orders=0
    number_of_uncompleted_orders=0
    total_distance_completed_orders=0
    total_distance_uncompleted_orders=0
    total_time_completed_orders=0
    total_time_uncompleted_orders=0
    for i, matrix in enumerate(distance_matrices):
        
        
        # Find the next available truck
        truck_idx = truck_availability.index(min(truck_availability))
        
        # Calculate the trip distance using the VRP algorithm
        trip_distance = ant_colony_vrp(matrix, alpha=1, beta=1, num_ants=10, evaporation_rate=0.5, num_iterations=100)
        trip_time = calculate_time(trip_distance,SPEED_KMH)
        
        # Check if the trip fits within the daily constraint of work hours for any truck
        if truck_availability[truck_idx] + trip_time <= WORK_HOURS_PER_DAY:
            # Assign the trip to the truck
            truck_availability[truck_idx] += trip_time
            total_distance += trip_distance
            number_completed_trips+=1
            number_of_completed_orders+=len(matrix)
            total_time_completed_orders+=trip_time
            total_distance_completed_orders+= trip_distance

        else:
            # trips that can't be delivered in the day
            number_uncompleted_trips+=1
            number_of_uncompleted_orders+=len(matrix)
            total_distance_uncompleted_orders+= trip_distance
            total_time_uncompleted_orders+=trip_time
            # Skip this trip but continue to process subsequent trips


    average_distance_for_successfull_orders=total_distance_completed_orders/number_completed_trips
    average_time_for_successfull_orders=total_time_completed_orders/number_completed_trips

    results=[number_completed_trips,number_uncompleted_trips,number_of_completed_orders,number_of_uncompleted_orders,total_distance_completed_orders,total_distance_uncompleted_orders,total_time_completed_orders,total_time_uncompleted_orders,average_distance_for_successfull_orders,average_time_for_successfull_orders]
    return results