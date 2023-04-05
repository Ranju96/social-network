#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx as nx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


import csv
input=pd.read_csv('Downloads\networkxDUU.txt')


# In[7]:


print(input)


# In[7]:


G_directed =nx.DiGraph()
G_directed.add_edges_from(zip(input['Source'],input['Destination']))

G_undirected =nx.Graph()
G_undirected.add_edges_from(zip(input['Source'],input['Destination']))


# In[8]:


fig,ax=plt.subplots(figsize=(10,10))
nx.draw_networkx(G_directed, with_labels=True,ax=ax)

plt.show()


# In[9]:


fig,ax=plt.subplots(figsize=(20,10))
nx.draw_networkx(G_undirected, with_labels=True)

plt.show()


# In[10]:


G_directed1 =nx.DiGraph()
edge_directed=list(zip(input['Source'],input['Destination'],input['Weight']))
G_directed1.add_weighted_edges_from(edge_directed)

G_undirected1 =nx.Graph()
edge_undirected=list(zip(input['Source'],input['Destination'],input['Weight']))
G_undirected1.add_weighted_edges_from(edge_undirected)


# In[11]:


#Draw directed Graph
plt.figure(figsize=(20,10))
nx.draw_networkx(G_directed1,arrows=True)
edge_labels = nx.get_edge_attributes(G_directed1,'weight')
nx.draw_networkx_edge_labels(G_directed1, pos=nx.spring_layout(G_directed1),edge_labels=edge_labels)
plt.show()


# In[12]:


#Draw undirected Graph
plt.figure(figsize=(20,10))
nx.draw_networkx(G_undirected1,arrows=False)
edge_labels = nx.get_edge_attributes(G_undirected1,'weight')
nx.draw_networkx_edge_labels(G_undirected1, pos=nx.spring_layout(G_undirected1),edge_labels=edge_labels)
plt.show()


# In[13]:


#n this example code, we first create an undirected graph with five nodes and six edges using the NetworkX library, and then draw the graph with edge labels using the draw_networkx() and draw_networkx_edge_labels() functions.

#Next, we use the degree() function to calculate the degree of each node in the graph and store the results in a dictionary using the dict() function. We then use the min() and max() functions to find the minimum and maximum degrees, respectively.

#Finally, we print the minimum and maximum degrees to the console using the print() function
# Draw the graph with node labels

G_undirected1 = nx.Graph()
G_undirected1.add_weighted_edges_from(edge_undirected)
plt.figure(figsize=(20,10))
# Draw the graph
nx.draw_networkx(G_undirected1,arrows=False)
edge_labels = nx.get_edge_attributes(G_undirected1,'weight')
nx.draw_networkx_edge_labels(G_undirected1, pos=nx.spring_layout(G_undirected1),edge_labels=edge_labels)

# Calculate the minimum and maximum degree of the nodes in the graph
degrees = dict(G_undirected1.degree())
min_degree = min(degrees.values())
max_degree = max(degrees.values())

# Display the minimum and maximum degree
print("Minimum degree:", min_degree)
print("Maximum degree:", max_degree)

plt.show()


# In[14]:


# Draw the graph
plt.figure(figsize=(15,10))
nx.draw_networkx(G_undirected1,arrows=False)
edge_labels = nx.get_edge_attributes(G_undirected1,'weight')
nx.draw_networkx_edge_labels(G_undirected1, pos=nx.spring_layout(G_undirected1),edge_labels=edge_labels)



# Calculate the minimum and maximum degree of the nodes in the graph
degrees = dict(G_undirected1.degree())
min_degree = min(degrees.values())
max_degree = max(degrees.values())

# Display the minimum and maximum degree
print("Minimum degree:", min_degree)
print("Maximum degree:", max_degree)

plt.show()


# In[15]:


plt.figure(figsize=(20,10))
G_directed1 = nx.Graph()
#G_directed1.add_weighted_edges_from(edge_directed)
# Draw the graph
nx.draw_networkx(G_directed1,arrows=True)
edge_labels = nx.get_edge_attributes(G_directed1,'weight')
nx.draw_networkx_edge_labels(G_directed1, pos=nx.spring_layout(G_directed1),edge_labels=edge_labels)


pos = nx.spring_layout(G_directed)
nx.draw_networkx(G_directed, pos=pos, with_labels=True)
edge_labels = nx.get_edge_attributes(G_directed, 'weight')
nx.draw_networkx_edge_labels(G_directed, pos=pos, edge_labels=edge_labels)

# Find the node with the maximum out-degree
out_degrees = dict(G_directed.out_degree())
max_out_degree = max(out_degrees.values())
max_out_node = max(out_degrees, key=out_degrees.get)
print("Node with maximum out-degree:", max_out_node)
print("Maximum out-degree:", max_out_degree)

# Find the node with the minimum out-degree
min_out_degree = min(out_degrees.values())
min_out_node = min(out_degrees, key=out_degrees.get)
print("Node with minimum out-degree:", min_out_node)
print("Minimum out-degree:", min_out_degree)


# Display the graph
plt.show()


# In[16]:


plt.figure(figsize=(20,10))
G_directed1 = nx.Graph()
G_directed1.add_weighted_edges_from(edge_directed)

# Draw the graph
nx.draw_networkx(G_directed1,arrows=True)
edge_labels = nx.get_edge_attributes(G_directed1,'weight')
nx.draw_networkx_edge_labels(G_directed1, pos=nx.spring_layout(G_directed1),edge_labels=edge_labels)

# Find the node with the maximum out-degree
if len(G_directed) > 0 and any(G_directed.out_degree()):
    out_degrees = dict(G_directed.out_degree())
    max_out_degree = max(out_degrees.values())
    max_out_node = max(out_degrees, key=out_degrees.get)
    print("Node with maximum out-degree:", max_out_node)
else:
    print("Graph is empty or all nodes have out-degree 0.")

# Find the node with the maximum in-degree
if len(G_directed) > 0 and any(G_directed.in_degree()):
    in_degrees = dict(G_directed.in_degree())
    max_in_degree = max(in_degrees.values())
    max_in_node = max(in_degrees, key=in_degrees.get)
    print("Node with maximum in-degree:", max_in_node)
else:
    print("Graph is empty or all nodes have in-degree 0.")

# Find the node with the minimum out-degree
if len(G_directed) > 0 and any(G_directed.out_degree()):
    out_degrees = dict(G_directed.out_degree())
    min_out_degree = min(out_degrees.values())
    min_out_node = min(out_degrees, key=out_degrees.get)
    print("Node with minimum out-degree:", min_out_node)
else:
    print("Graph is empty or all nodes have out-degree 0.")

# Find the node with the minimum in-degree
if len(G_directed) > 0 and any(G_directed.in_degree()):
    in_degrees = dict(G_directed.in_degree())
    min_in_degree = min(in_degrees.values())
    min_in_node = min(in_degrees, key=in_degrees.get)
    print("Node with minimum in-degree:", min_in_node)
else:
    print("Graph is empty or all nodes have in-degree 0.")


# In[17]:


#This code creates an undirected graph with 4 nodes and #4 edges using the Graph() function from NetworkX. We then use the to_numpy_matrix() function to create the adjacency matrix, and print it to the console.

#To calculate the row sums of the adjacency matrix, we use the numpy.sum() function with the axis parameter set to 1 to sum across the rows. We then print the row sums to the console.

G_undirected1 = nx.Graph()
G_undirected1.add_weighted_edges_from(edge_undirected)

# Create the adjacency matrix
A = nx.to_numpy_matrix(G_undirected1)

# Print the adjacency matrix
print("Adjacency matrix:")
print(A)

# Print the row sums
row_sums = np.sum(A, axis=1)
print("Row sums:")
print(row_sums)


# In[18]:


# Create a directed graph
#G_undirected1 = nx.Graph()
#G_undirected1.add_weighted_edges_from(edge_undirected)

G_directed1 =nx.DiGraph()
edge_directed=list(zip(input['Source'],input['Destination'],input['Weight']))
G_directed1.add_weighted_edges_from(edge_directed)

# Create the adjacency matrix
A = nx.to_numpy_matrix(G_directed, weight=None)

# Print the adjacency matrix
print("Adjacency matrix:")
print(A)

# Print the row sums and column sums for node 2
node = 2
row_sum = np.sum(A[node, :])
col_sum = np.sum(A[:, node])

print("Row sum for node", node, ":", row_sum)
print("Column sum for node", node, ":", col_sum)


# In[19]:


#for undirected graph
# degree centrality
degree_centrality = nx.degree_centrality(G_undirected1)
print("Degree Centrality:")
print(degree_centrality)

# closeness centrality
closeness_centrality = nx.closeness_centrality(G_undirected1)
print("\nCloseness Centrality:")
print(closeness_centrality)

# betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G_undirected1)
print("\nBetweenness Centrality:")
print(betweenness_centrality)

# eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(G_undirected1)
print("\nEigenvector Centrality:")
print(eigenvector_centrality)


# In[20]:


# calculate PageRank
pagerank = nx.pagerank(G_undirected1)
print("PageRank:")
print(pagerank)


# In[21]:


#Centrality Measures for directed Graph
# degree centrality
in_degree_centrality = nx.in_degree_centrality(G_directed1)
out_degree_centrality = nx.out_degree_centrality(G_directed1)
print("In-Degree Centrality:")
print(in_degree_centrality)
print("\nOut-Degree Centrality:")
print(out_degree_centrality)

# closeness centrality
closeness_centrality = nx.closeness_centrality(G_directed1)
print("\nCloseness Centrality:")
print(closeness_centrality)

# betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G_directed1)
print("\nBetweenness Centrality:")
print(betweenness_centrality)

# eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(G_directed1)
print("\nEigenvector Centrality:")
print(eigenvector_centrality)


# In[22]:


# PageRank
pagerank = nx.pagerank(G_directed1)
print("\nPageRank:")
print(pagerank)


# In[23]:


# calculate centrality measures
# degree centrality
degree_centrality = nx.degree_centrality(G_undirected1)
closeness_centrality = nx.closeness_centrality(G_undirected1)
betweenness_centrality = nx.betweenness_centrality(G_undirected1)
pagerank = nx.pagerank(G_undirected1)

# find minimum and maximum centrality scores and their corresponding nodes
min_degree_node = min(degree_centrality,key=degree_centrality.get)
max_degree_node = max(degree_centrality,key=degree_centrality.get)
min_closeness_node = min(closeness_centrality, key=closeness_centrality.get)
max_closeness_node = max(closeness_centrality, key=closeness_centrality.get)
min_betweenness_node = min(betweenness_centrality, key=betweenness_centrality.get)
max_betweenness_node = max(betweenness_centrality, key=betweenness_centrality.get)
min_pagerank_node = min(pagerank, key=pagerank.get)
max_pagerank_node = max(pagerank, key=pagerank.get)

min_degree_score = degree_centrality[min_degree_node]
max_degree_score = degree_centrality[max_degree_node]
min_closeness_score = closeness_centrality[min_closeness_node]
max_closeness_score = closeness_centrality[max_closeness_node]
min_betweenness_score = betweenness_centrality[min_betweenness_node]
max_betweenness_score = betweenness_centrality[max_betweenness_node]
min_pagerank_score = pagerank[min_pagerank_node]
max_pagerank_score = pagerank[max_pagerank_node]

# print results
print("Degree Centrality:")
print("Minimum score:", min_degree_score, "for node", min_degree_node)
print("Maximum score:", max_degree_score, "for node", max_degree_node)

print("\nCloseness Centrality:")
print("Minimum score:", min_closeness_score, "for node", min_closeness_node)
print("Maximum score:", max_closeness_score, "for node", max_closeness_node)

print("\nBetweenness Centrality:")
print("Minimum score:", min_betweenness_score, "for node", min_betweenness_node)
print("Maximum score:", max_betweenness_score, "for node", max_betweenness_node)

print("\nPageRank:")
print("Minimum score:", min_pagerank_score, "for node", min_pagerank_node)
print("Maximum score:", max_pagerank_score, "for node", max_pagerank_node)


# In[24]:


def dict_round(s_r_dict):
  s_r_dict = dict(sorted(s_r_dict.items()))
  for key in s_r_dict:
    s_r_dict[key] = round(s_r_dict[key], 2)
  return s_r_dict


# In[ ]:


deg_centrality = nx.degree_centrality(G)
print(deg_centrality)
udg_deg_centrality = dict_round(deg_centrality)
print(udg_deg_centrality)


# In[ ]:


closeness_centrality = nx.closeness_centrality(G)
print(closeness_centrality)
udg_closeness_centrality = dict_round(closeness_centrality)
print(udg_closeness_centrality)


# In[ ]:


betweeness_centrality = nx.closeness_centrality(G)
print(betweeness_centrality)
udg_betweeness_centrality = dict_round(betweeness_centrality)
print(udg_betweeness_centrality)


# In[ ]:


eigen_vector_centrality = nx.eigenvector_centrality(G)
print(eigen_vector_centrality)
udg_eigen_vector_centrality = dict_round(eigen_vector_centrality)
print(udg_eigen_vector_centrality)


# In[ ]:


udf = pd.DataFrame.from_dict(udg_deg_centrality,orient='index',columns=['Degree_Centrality'])
udf = udf.assign(Closeness_Centrality= udg_closeness_centrality.values())
udf = udf.assign(Betweeness_Centrality= udg_betweeness_centrality.values())
udf = udf.assign(Eigen_Vector_Centrality= udg_eigen_vector_centrality.values())


# In[ ]:


udf


# In[ ]:


udf.to_csv('All Centality Degrees (Undirected Graph).csv')


# In[ ]:


def find_node_min_centarlity(udf,cent_name):
  min = 100
  node_min = -1
  for i in range(len(udf)):
    if udf[cent_name][i] < min:
      min =  udf[cent_name][i]
      node_min = i
  return node_min,min

def find_node_max_centarlity(udf,cent_name):
  max = 0
  node_max = -1
  for i in range(len(udf)):
    if udf[cent_name][i] > max:
      max =  udf[cent_name][i]
      node_max = i
  return node_max,max


# In[ ]:


node_min,min = find_node_min_centarlity(udf,'Degree_Centrality')
node_max,max = find_node_max_centarlity(udf,'Degree_Centrality')
print(node_min,min)
print(node_max,max)
degree_cent_min_max = {'Min Degree Node':str(node_min),'Value C1':min,'Max Degree Node':str(node_max),'Value C2':max}


# In[ ]:


node_min,min = find_node_min_centarlity(udf,'Closeness_Centrality')
node_max,max = find_node_max_centarlity(udf,'Closeness_Centrality')
print(node_min,min)
print(node_max,max)
closeness_cent_min_max = {'Min Degree Node':str(node_min),'Value C1':min,'Max Degree Node':str(node_max),'Value C2':max}


# In[ ]:


node_min,min = find_node_min_centarlity(udf,'Betweeness_Centrality')
node_max,max = find_node_max_centarlity(udf,'Betweeness_Centrality')
print(node_min,min)
print(node_max,max)
betweeness_cent_min_max = {'Min Degree Node':str(node_min),'Value C1':min,'Max Degree Node':str(node_max),'Value C2':max}


# In[ ]:


node_min,min = find_node_min_centarlity(udf,'Eigen_Vector_Centrality')
node_max,max = find_node_max_centarlity(udf,'Eigen_Vector_Centrality')
print(node_min,min)
print(node_max,max)
eigen_vector_cent_min_max = {'Min Degree Node':str(node_min),'Value C1':min,'Max Degree Node':str(node_max),'Value C2':max}


# In[ ]:


udf_min_max = pd.DataFrame(columns=['Min Degree Node','Value C1','Max Degree Node','Value C2'])

udf_min_max


# In[ ]:


udf_min_max = udf_min_max.append(pd.Series(degree_cent_min_max,index = udf_min_max.columns,name = 'Degree Centrality'))
udf_min_max = udf_min_max.append(pd.Series(closeness_cent_min_max,index = udf_min_max.columns,name = 'Closeness Centrality'))
udf_min_max = udf_min_max.append(pd.Series(betweeness_cent_min_max,index = udf_min_max.columns,name = 'Betweeness Centrality'))
udf_min_max = udf_min_max.append(pd.Series(eigen_vector_cent_min_max,index = udf_min_max.columns,name = 'Eigen Vector Centrality'))
udf_min_max


# In[ ]:


#directed graph
deg_centrality = nx.degree_centrality(G2)
print(deg_centrality)
dg_deg_centrality = dict_round(deg_centrality)
print(dg_deg_centrality)


# In[ ]:


closeness_centrality = nx.closeness_centrality(G2)
print(closeness_centrality)
dg_closeness_centrality = dict_round(closeness_centrality)
print(dg_closeness_centrality)


# In[ ]:


betweeness_centrality = nx.closeness_centrality(G2)
print(betweeness_centrality)
dg_betweeness_centrality = dict_round(betweeness_centrality)
print(dg_betweeness_centrality)


# In[ ]:


ddf = pd.DataFrame.from_dict(dg_deg_centrality,orient='index',columns=['Degree_Centrality'])
ddf = ddf.assign(Closeness_Centrality= dg_closeness_centrality.values())
ddf = ddf.assign(Betweeness_Centrality= dg_betweeness_centrality.values())
#ddf = ddf.assign(Eigen_Vector_Centrality= dg_eigen_vector_centrality.values())
ddf


# In[ ]:


node_min,min = find_node_min_centarlity(udf,'Degree_Centrality')
node_max,max = find_node_max_centarlity(udf,'Degree_Centrality')
print(node_min,min)
print(node_max,max)
degree_cent_min_max = {'Min Degree Node':str(node_min),'Value C1':min,'Max Degree Node':str(node_max),'Value C2':max}


# In[ ]:


node_min,min = find_node_min_centarlity(ddf,'Closeness_Centrality')
node_max,max = find_node_max_centarlity(ddf,'Closeness_Centrality')
print(node_min,min)
print(node_max,max)
closeness_cent_min_max = {'Min Degree Node':str(node_min),'Value C1':min,'Max Degree Node':str(node_max),'Value C2':max}


# In[ ]:


node_min,min = find_node_min_centarlity(ddf,'Betweeness_Centrality')
node_max,max = find_node_max_centarlity(ddf,'Betweeness_Centrality')
print(node_min,min)
print(node_max,max)
betweeness_cent_min_max = {'Min Degree Node':str(node_min),'Value C1':min,'Max Degree Node':str(node_max),'Value C2':max}


# In[ ]:


ddf_min_max = pd.DataFrame(columns=['Min Degree Node','Value C1','Max Degree Node','Value C2'])

ddf_min_max


# In[ ]:


ddf_min_max = ddf_min_max.append(pd.Series(degree_cent_min_max,index = ddf_min_max.columns,name = 'Degree Centrality'))
ddf_min_max = ddf_min_max.append(pd.Series(closeness_cent_min_max,index = ddf_min_max.columns,name = 'Closeness Centrality'))
ddf_min_max = ddf_min_max.append(pd.Series(betweeness_cent_min_max,index = ddf_min_max.columns,name = 'Betweeness Centrality'))
#ddf_min_max = ddf_min_max.append(pd.Series(eigen_vector_cent_min_max,index = ddf_min_max.columns,name = 'Eigen Vector Centrality'))
ddf_min_max

