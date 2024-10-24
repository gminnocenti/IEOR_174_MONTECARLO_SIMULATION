
from scipy.stats import gaussian_kde
import numpy as np
import math
import time

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

    