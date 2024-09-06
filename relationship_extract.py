from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
import pickle
import random
import os

df = pd.read_csv('category_structure.csv', header = 2)
categories = df[~df['L2'].isna()]['L2'].unique()


# Only keep the randomly sampled 1000 source product categories
full_list = []
for ind, first_type in enumerate(categories):   
    for second_type in categories[ind+1:]:
        full_list.append((first_type, second_type))
partial_list = random.sample(full_list, 1000)


# Generate scenarios for each source product category
scenario_data = {}
for first_type, second_type in tqdm(partial_list_2):
    # Check if first_type is already in scenario_data
    if first_type not in scenario_data:
        # Run simulate_scenarios and store the result if first_type is not already filled
        scenarios = simulate_scenarios(first_type)
        scenario_data[first_type] = {
            "scenarios": scenarios
        }
    else:
        # Optionally, you can handle the case where first_type is already filled
        # or just continue with the loop.
        continue

# Save relationship_counts
with open('scenario_data.pkl', 'wb') as f:
    pickle.dump(scenario_data, f)


relationship_data = {}
for first_type, second_type in tqdm(partial_list_2):
    scenarios = scenario_data[first_type]
    sub_count = 0
    comp_count = 0
    irre_count = 0

    for i, scenario in enumerate(scenarios['scenarios']):
        label = 'hi'
        label = get_relationship(scenario, first_type, second_type)
        
        sub_count += (label == 'substitute')
        comp_count += (label == 'complementary')
        irre_count += (label == 'irrelevant')

        time.sleep(1)
        
    relationship_data[(first_type, second_type)] = {
        "substitute": sub_count/5,
        "complementary": comp_count/5,
        "irrelevant": irre_count/5,
    }

# Save relationship_counts
with open('relationship_data.pkl', 'wb') as f:
    pickle.dump(relationship_data, f)

# Record data and relationships
unique_lis_from = []
unique_lis_to = []
unique_lis_sub = []
unique_lis_comp = []
for i in relationship_data:
    unique_lis_from.append(i[0])
    unique_lis_to.append(i[1])
    unique_lis_sub.append(relationship_data[i]['substitute'])
    unique_lis_comp.append(relationship_data[i]['complementary'])


# Construct the substitue relationship knowledge graphs
df1 = pd.DataFrame(columns=["From Type", "From Name", "Edge Type", "To Type", "To Name", "Weight"])
df1["From Name"] = unique_lis_from
df1["To Name"] = unique_lis_to
df1["From Type"] = 'L2 Category'
df1["To Type"] = 'L2 Category'
df1["Edge Type"] = 'can be substituted by'
df1["Weight"] = unique_lis_sub
df1.to_csv('sub_kg_v1.csv')


# Construct the complementary relationship knowledge graphs
df2 = pd.DataFrame(columns=["From Type", "From Name", "Edge Type", "To Type", "To Name", "Weight"])
df2["From Name"] = unique_lis_from
df2["To Name"] = unique_lis_to
df2["From Type"] = 'L2 Category'
df2["To Type"] = 'L2 Category'
df2["Edge Type"] = 'can be complemented by'
df2["Weight"] = unique_lis_comp
df2.to_csv('comp_kg_v1.csv')


# Some ways to visualize the matrix correlations
# Create an adjacency matrix
# Pivot the DataFrame to get a matrix form
adj_matrix = df2.pivot(index='From Name', columns='To Name', values='Weight').fillna(0)

# Convert the matrix to a DataFrame if it's not already
adj_matrix = pd.DataFrame(adj_matrix)

# Filter out the zero-weight entries (optional)
filtered_matrix = adj_matrix.replace(0, np.nan).dropna(how='all', axis=1).dropna(how='all', axis=0)

# Stack the matrix and reset the index to get a long format DataFrame
adjacency_entries = filtered_matrix.stack().reset_index()
adjacency_entries.columns = ['From Node', 'To Node', 'Weight']

# Sort by 'Weight' in ascending order
sorted_adjacency_entries = adjacency_entries.sort_values(by='Weight', ascending=False)
sorted_adjacency_entries[sorted_adjacency_entries['Weight']<0.3].head(30)


# Find the shortest path distance between product categories
# Create an empty graph
G = nx.Graph()

# Invert the weights and add edges to the graph
for index, row in df.iterrows():
    if row['Weight'] > 0:
        # Inverting the weight
        inverted_weight = 1 / row['Weight']
        G.add_edge(row['From Name'], row['To Name'], weight=inverted_weight)
    else:
        # If weight is 0, treat it as an infinite distance (no relationship)
        G.add_edge(row['From Name'], row['To Name'], weight=float('inf'))

# Calculate the shortest path distance between all pairs of nodes
shortest_path_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

# Convert the shortest path distances to a DataFrame for easy visualization
shortest_path_df = pd.DataFrame(shortest_path_dict).fillna(float('inf'))

