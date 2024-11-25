<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monte Carlo Simulation for Distribution Center Optimization</title>
</head>
<body>
    <h1>Monte Carlo Simulation for Distribution Center Optimization</h1>
    
    <h2>Overview</h2>
    <p>
        This repository contains a Monte Carlo simulation designed to help an e-commerce store 
        determine the optimal location for a new distribution center in São Paulo, Brazil. The simulation 
        leverages historical order data from the 
        <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_sellers_dataset.csv" target="_blank">
            Brazilian E-Commerce Dataset on Kaggle
        </a> 
        to model daily operations and evaluate the efficiency of two potential locations.
    </p>
    <p>The simulation models:</p>
    <ol>
        <li>Daily order generation.</li>
        <li>Product assignment to orders.</li>
        <li>Delivery address selection from a pool of top 200 high-frequency customers.</li>
        <li>Truck loading based on product volume.</li>
        <li>
            Route optimization using an Ant Colony Optimization (ACO) approach for the Vehicle Routing 
            Problem (VRP), with distance calculations and working hour constraints.
        </li>
    </ol>

    <h2>Key Features</h2>
    <h3>1. Order Simulation</h3>
    <ul>
        <li>Simulates the number of orders to deliver in a day.</li>
        <li>Assigns a random number of products to each order.</li>
        <li>Randomly selects products for each order from historical data.</li>
    </ul>
    <h3>2. Address Simulation</h3>
    <ul>
        <li>Simulates delivery addresses from the top 200 historical customers who ordered the most.</li>
    </ul>
    <h3>3. Truck Loading Algorithm</h3>
    <ul>
        <li>Optimizes how orders are loaded into trucks based on product volume.</li>
        <li>Estimates the number of trips required to deliver all orders in a day.</li>
    </ul>
    <h3>4. Route Optimization</h3>
    <ul>
        <li>Implements an Ant Colony Optimization algorithm for the VRP.</li>
        <li>
            Calculates the total distance and evaluates the feasibility of trips with 2 or 3 trucks 
            and an 8-hour daily work limit.
        </li>
        <li>Stores results after each iteration for analysis.</li>
    </ul>

    <h2>Dataset</h2>
    <p>
        The simulation uses the 
        <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_sellers_dataset.csv" target="_blank">
            Brazilian E-Commerce Dataset
        </a>, 
        specifically focusing on the <code>olist_sellers_dataset.csv</code> file to derive insights for simulation.
    </p>

    <h2>Setup Instructions</h2>
    <ol>
        <li>Clone this repository:
            <pre><code>git clone https://github.com/yourusername/montecarlo-distribution-center.git
cd montecarlo-distribution-center
            </code></pre>
        </li>
        <li>Install the required Python libraries:
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li>Run the simulation:
            <pre><code>python simulation_main.py</code></pre>
        </li>
    </ol>

    <h2>File Structure</h2>
    <ul>
        <li><code>simulation_main.py</code>: Main script to execute the Monte Carlo simulation.</li>
        <li><code>functions_for_simulation.py</code>: Contains reusable functions used in the simulation 
            (e.g., order generation, truck loading, and route optimization).
        </li>
        <li><code>requirements.txt</code>: Lists the Python libraries required to run the simulation.</li>
        <li><code>README.md</code>: Documentation for the repository.</li>
    </ul>

    <h2>Customization</h2>
    <p>
        To explore or modify specific aspects of the simulation, refer to 
        <code>functions_for_simulation.py</code>. This file includes detailed implementations for each step 
        of the simulation.
    </p>

    <h2>Results</h2>
    <p>The simulation outputs:</p>
    <ul>
        <li>The number of trips required for each scenario.</li>
        <li>Total distance traveled by trucks.</li>
        <li>Comparative performance metrics for the two candidate locations.</li>
    </ul>
    <p>
        These results help identify the most efficient location for the new distribution center 
        based on daily operations.
    </p>

    <p>Contributions and feedback are welcome! If you encounter issues or have suggestions, feel free to open an issue or submit a pull request. 🚀</p>
</body>
</html>
