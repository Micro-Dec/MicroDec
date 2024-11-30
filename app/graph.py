from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import numpy as np
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import warnings 
import collections
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import pandas as pd
from stellargraph import StellarGraph
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from transformers import BertTokenizer, BertModel
import torch
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaModel


def create_graph(nodes_file, dependencies_file, type, appName):
    # Load citation data
    cora_cites_file = dependencies_file
    cora_cites = pd.read_csv(
        cora_cites_file,
        sep=",",
        header=0,
        names=["target", "source"],
    )
    cora_cites

    # Load content data
    cora_content_file = nodes_file
    cora_feature_names = [f"w{i}" for i in range(len(pd.read_csv(nodes_file).columns) - 1)]
    cora_raw_content = pd.read_csv(
        cora_content_file,
        sep=",",
        header=0,
        names=["id", *cora_feature_names],
    )
    cora_raw_content

    # Set index for content data
    cora_content_str_subject = cora_raw_content.set_index("id")
    cora_content_str_subject

    #Define jaccard_weights function
    def jaccard_weights(graph, _subjects, edges):
        sources = graph.node_features(edges["source"])
        targets = graph.node_features(edges["target"])

        intersection = np.logical_and(sources, targets)
        union = np.logical_or(sources, targets)

        return intersection.sum(axis=1) / union.sum(axis=1)

    G = StellarGraph({"paper": cora_content_str_subject}, {"cites": cora_cites})

    # Apply jaccard_weights to the graph edges
    edge_weights = jaccard_weights(G, None, cora_cites)

    cora_cites['weight'] = edge_weights

    # Normalize the weights
    scaler = MinMaxScaler()
    cora_cites['weight'] = scaler.fit_transform(cora_cites[['weight']])

    # Create the graph with weighted edges
    G = StellarGraph({"paper": cora_content_str_subject}, {"cites": cora_cites[['source', 'target', 'weight']]})

    G_nx = G.to_networkx()
    isolated_nodes = list(nx.isolates(G_nx))
    G_nx.remove_nodes_from(isolated_nodes)
    remaining_nodes = list(G_nx.nodes())
    remaining_features = cora_content_str_subject.loc[remaining_nodes]
    # Convert back to StellarGraph
    G = StellarGraph({"paper": remaining_features}, {"cites": nx.to_pandas_edgelist(G_nx)[['source', 'target', 'weight']]})
    num_nodes = G.number_of_nodes()
    weighted_walks = random_walks(G, num_nodes)

    
    
    if type == "bert":    
        node_ids, weighted_node_embeddings = embeddings (weighted_walks)
    elif type == "openai":
       node_ids, weighted_node_embeddings =  embeddings_openai(weighted_walks,appName)
    

    return node_ids, weighted_node_embeddings, num_nodes

from sklearn.linear_model import LinearRegression

def determine_random_walk_parameters(num_nodes):
    # Empirical data
    node_counts = np.array([15, 84]).reshape(-1, 1)
    lengths = np.array([20, 100])
    walks = np.array([15, 10])

    # Fit linear models
    length_model = LinearRegression().fit(node_counts, lengths)
    walk_model = LinearRegression().fit(node_counts, walks)

    #predict length and number of walks based on the linear models
    length = int(round(length_model.predict([[num_nodes]])[0]))
    number_of_random_walks = int(round(walk_model.predict([[num_nodes]])[0]))
    
    #ensure minimum values to avoid too short walks 
    #length = max(length, 20)
    length = max(min(length, 100), 20)  # Ensures length is between 20 and 100
    number_of_random_walks = max(number_of_random_walks, 10)
    
    return length, number_of_random_walks

def random_walks(G, num_nodes):
#def random_walks(G, length, number_of_random_walks):
    
    length, number_of_random_walks = determine_random_walk_parameters(num_nodes)



    rw = BiasedRandomWalk(G)
    weighted_walks = rw.run(
    nodes=G.nodes(),  # root nodes
    length= length,  # maximum length of a random walk
    n= number_of_random_walks, # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2,  # Defines (unormalised) probability, 1/q, for moving away from source node
    weighted=True,  # for weighted random walks
    seed=42,  # random seed fixed for reproducibility
    )

    return weighted_walks


def embeddings (weighted_walks):
    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
   
    # Convert random walks to strings
    walks_as_strings = [' '.join(map(str, walk)) for walk in weighted_walks]

    # Initialize dictionary for embeddings and a set for unique node IDs
    node_embeddings = defaultdict(list)
    node_id_set = set()

    # Process each walk
    for walk in walks_as_strings:
        inputs = tokenizer(walk, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embedding as the sentence representation
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        nodes = walk.split()
        for node in nodes:
            node_embeddings[node].append(embedding)
        node_id_set.update(nodes)

    # Average embeddings for each node
    final_node_embeddings = {node: np.mean(embeddings, axis=0) for node, embeddings in node_embeddings.items()}

    # Convert to list and array
    node_ids = list(final_node_embeddings.keys())
    weighted_node_embeddings = np.array(list(final_node_embeddings.values()))


    return node_ids, weighted_node_embeddings


def embeddings_openai(weighted_walks, appName):
    import openai
    import numpy as np
    from collections import defaultdict
    import os
    import pickle
    openai.api_key = '####'
    embedding_file_path = os.path.join('data/openAIembeddings', appName)
    # check if the embeddings exist
    if os.path.exists(embedding_file_path):
        with open(embedding_file_path, 'rb') as f:
            node_ids, weighted_node_embeddings = pickle.load(f)
        return node_ids, weighted_node_embeddings

    def get_openai_embedding(text):
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return np.array(response['data'][0]['embedding'])

    # Convert random walks to strings
    walks_as_strings = [' '.join(map(str, walk)) for walk in weighted_walks]

    
    node_embeddings = defaultdict(list)
    node_id_set = set()

    #process each walk
    for walk in walks_as_strings:
        embedding = get_openai_embedding(walk)
        
        nodes = walk.split()
        for node in nodes:
            node_embeddings[node].append(embedding)
        node_id_set.update(nodes)

    # Average embeddings
    final_node_embeddings = {node: np.mean(embeddings, axis=0) for node, embeddings in node_embeddings.items()}

    #convert to list and array
    node_ids = list(final_node_embeddings.keys())
    weighted_node_embeddings = np.array(list(final_node_embeddings.values()))

    #save the embeddings 
    os.makedirs('data/openAIembeddings', exist_ok=True)
    with open(embedding_file_path, 'wb') as f:
        pickle.dump((node_ids, weighted_node_embeddings), f)

    return node_ids, weighted_node_embeddings


  