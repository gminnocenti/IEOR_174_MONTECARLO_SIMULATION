import pandas as pd
import numpy as np
from functions_for_simulation import simulate_products_all_orders,rejection_sampling,generate_matrix_simulation,simulate_number_of_products_in_a_order,simulate_number_of_orders_in_a_day
#Lodad Datasets
df_number_of_orders_to_deliver=pd.read_csv("SIM_density_number_of_orders_delivered_in_a_day.csv")
df_number_of_products_in_one_order=pd.read_csv("SIM_number_of_products_in_a_order.csv")
df_most_ordered_products=pd.read_csv("SIM_most_ordered_products.csv")

distance_matrix_distribution_center1 = np.loadtxt("SIM_distribution_center_1_distance_matrix.csv", delimiter=",")
distance_matrix_distribution_center2 = np.loadtxt("SIM_distribution_center_2_distance_matrix.csv", delimiter=",")

#First simulate the number of orders that will be delivered in the day
n_orders=simulate_number_of_orders_in_a_day(df_number_of_orders_to_deliver['Number_of_orders_delivered'].values,df_number_of_orders_to_deliver['count'].values,1)


#Second Simulate the number of products in each order
size_of_orders=simulate_number_of_products_in_a_order( df_number_of_products_in_one_order['Order_size_count'].values,df_number_of_products_in_one_order['count'].values,int(n_orders[0]))

# Third Simulate the products in each order
product_id_for_each_order=simulate_products_all_orders(df_most_ordered_products,size_of_orders)

#Fourth simulate the houses that made the orders
distance_matrix_simulation_DC_1,distance_matrix_simulation_DC_2=generate_matrix_simulation(distance_matrix_distribution_center1,distance_matrix_distribution_center2,n_orders[0])




print(f"Number of orders to deliver{n_orders}")
print(f"Number of products in each order {size_of_orders}")
print("##############################################################")
for i in range(len(size_of_orders)):
    print(f"Order # {i} -> Order Size: {size_of_orders[i]}")
    print(f'Order Size: {size_of_orders[i]}')
    print(f'Product Ids = {product_id_for_each_order[i]}')
  
    print("____________________________________________________________")