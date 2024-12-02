# Monte Carlo Simulation for Distribution Center Optimization

## Overview

This repository contains a Monte Carlo simulation designed to help an e-commerce store determine the optimal location for a new distribution center in São Paulo, Brazil. The simulation leverages historical order data from the Brazilian E-Commerce Dataset on Kaggle to model daily operations and evaluate the efficiency of two potential locations.

The simulation models:

1. Daily order generation.
2. Product assignment to orders.
3. Delivery address selection from a pool of top 200 high-frequency customers.
4. Truck loading based on product volume.
5. Route optimization using an Ant Colony Optimization (ACO) approach for the Vehicle Routing Problem (VRP), with distance calculations and working hour constraints.

## Key Features

### 1. Order Simulation
- Simulates the number of orders to deliver in a day.
- Assigns a random number of products to each order.
- Simulates products for each order from historical data. For this, we used the Rejection Sampling technique based on the historical data and the Probability Density Function (PDF) for each step.


### 2. Address Simulation
- Simulates delivery addresses from the top 200 historical customers who ordered the most.

### 3. Truck Loading Algorithm
- Optimizes how orders are loaded into trucks based on product volume.
- Estimates the number of trips required to deliver all orders in a day.

### 4. Route Optimization
- Implements an Ant Colony Optimization algorithm for the VRP.
- Calculates the total distance and evaluates the feasibility of trips with 2 or 3 trucks and an 8-hour daily work limit.
- Stores results after each iteration for analysis.

## Dataset
The simulation uses the [Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_sellers_dataset.csv), specifically focusing on the `olist_sellers_dataset.csv` file to derive insights for simulation.


## Setup Instructions

1. Clone this repository:
```
git clone https://github.com/gminnocenti/IEOR_174_MONTECARLO_SIMULATION.git
```
2. Install the required Python libraries in your virtual environment:
```
   pip install -r requirements.txt
```

3. Run the simulation:
```
python simulation_main.py
```
## File Structure

- simulation_main.py: Main script to execute the Monte Carlo simulation.
- functions_for_simulation.py: Contains reusable functions used in the simulation such as order generation, truck loading, and route optimization.
- requirements.txt: Lists the Python libraries required to run the simulation.
- README.md: Documentation for the repository.

## Customization

To explore or modify specific aspects of the simulation, refer to functions_for_simulation.py. This file includes detailed implementations for each step of the simulation.

## Results

The simulation outputs the following:

- The number of trips required for each scenario.
- Total distance traveled by trucks.
- Comparative performance metrics for the two candidate locations.

These results help identify the most efficient location for the new distribution center based on daily operations.

Contributions and feedback are welcome. .
