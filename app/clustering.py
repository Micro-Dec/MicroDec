from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import json
import csv
from sklearn.decomposition import PCA
import metrics.SMQ as SMQ
import metrics.CMQ as CMQ
import metrics.CHM as CHM
import metrics.CHD as CHD
import metrics.IFN as IFN
from data_preprocessing import update_output
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import subprocess
import warnings
from sklearn.exceptions import ConvergenceWarning

def clustring(node_ids,weighted_node_embeddings,appName, num_classes):
    metrics_path = 'projects.json'
    output_path_csv = '/microDec/app/metrics/output_fosci.csv'
    directory="/microDec"
    project_path = f'/microDec/apps/{appName}/src/main/java'
    sse = []
    silhouette_scores = []
    metrics_results = {}
    # determine max_n_clusters with a cap of 80
    if num_classes > 80:
        max_n_clusters = 80
    else:
        max_n_clusters = num_classes

    # Generate the range of clusters
    range_n_clusters = list(range(5, max_n_clusters)) # without +1
    #range_n_clusters = list(range(5, max_n_clusters))


    def load_data(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)


    def save_data(file_path, data):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)


    def update_cluster_string(metrics_path, clusters):
        data = load_data(metrics_path)
        cluster_string = {int(cluster): clusters[cluster] for cluster in clusters}
        data[0]['clusterString'] = str(cluster_string)
        save_data(metrics_path, data)


    def update_csv_with_clusters(csv_path, named_clusters):
        class_to_cluster = {}
        for cluster_id, classes in named_clusters.items():
            for class_name in classes:
                class_to_cluster[class_name] = cluster_id

        updated_rows = []
        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                class_name = row[1]
                if class_name in class_to_cluster:
                    row[0] = class_to_cluster[class_name]
                updated_rows.append(row)

        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(updated_rows)

    #NEW updata
    def run_java_command(DIRECTORY,PROJECT_PATH):
    
        try:
            output = subprocess.check_output(
                f"java -Dmetrics -Dproject={PROJECT_PATH} -cp symbolsolver-1.0.jar Main",
                cwd=f"{DIRECTORY}/symbolsolver/target/", 
                shell=True
            )
            output = str(output).replace("b'", "").split("\\n")
            print(output)
        except subprocess.CalledProcessError as e:
            print("Failed to execute command:", e)
            print("Output:", e.output.decode())        

    # Function to calculate metrics
    def calculate_metrics(output_path_csv, metrics_path, named_clusters):
        update_cluster_string(metrics_path, named_clusters)
        
        #run_java_command(directory,project_path)
        update_csv_with_clusters(output_path_csv, named_clusters)
        update_output()

        chm_value = CHM.calculate(output_path_csv)
        chd_value = CHD.calculate(output_path_csv)
        ifn_value = IFN.calculate(output_path_csv)
        smq_value, scoh_value, scop_value = SMQ.calculateWrapper()
        cmq_value, ccoh_value, ccop_value = CMQ.calculateWrapper()
        
        return {
            'CHM': chm_value,
            'CHD': chd_value,
            'IFN': ifn_value,
            'SMQ': smq_value,
            'scoh': scoh_value,
            'scop': scop_value,
            'CMQ': cmq_value,
            'ccoh': ccoh_value,
            'ccop': ccop_value,
        }


    # Normalize embeddings
    scaler = StandardScaler()
    weighted_node_embeddings = scaler.fit_transform(weighted_node_embeddings)


    
    def run_clustering_and_evaluate(algorithm_name, clustering_algorithm, range_n_clusters, weighted_node_embeddings):
        

        for n_clusters in range_n_clusters:
            clustering_algorithm.set_params(n_clusters=n_clusters)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    cluster_labels = clustering_algorithm.fit_predict(weighted_node_embeddings)
                if len(set(cluster_labels)) < n_clusters:
                    print(f"Warning: Only {len(set(cluster_labels))} distinct clusters found for n_clusters={n_clusters}")
                    continue
                clusters = {i+1: [] for i in range(n_clusters)}  
                for node_id, cluster_label in zip(node_ids, cluster_labels):
                    clusters[cluster_label + 1].append(node_id) 
        
                output_json_file = "/microDec/data/output.json"
                with open(output_json_file, 'r') as f:
                    class_data = json.load(f)

                #mapping from node id to actual names
                id_to_name = {str(i): key for i, key in enumerate(class_data.keys(), 1)}  # Start enumeration from 1

                # Link node id with actual names 
                named_clusters = {cluster: [id_to_name[node_id] for node_id in node_ids] for cluster, node_ids in clusters.items()}
                
                # Calculate and store metrics
                metrics_result = calculate_metrics(output_path_csv, metrics_path, named_clusters)
            
                metrics_results[f"{algorithm_name}_{n_clusters}"] = {
                'metrics': metrics_result,
                'named_clusters': named_clusters
                }
            except Exception as e:
                    print(f"Error processing clustering for n_clusters={n_clusters}: {e}")
                    continue    
    #K-means
    kmeans = KMeans(init='k-means++', n_init=10, random_state=123)
    run_clustering_and_evaluate('KMeans', kmeans, range_n_clusters, weighted_node_embeddings)

   
    with open(f'data/all_results_with_scoh_OpenAI/metrics_results_{appName}.json', 'w') as f:
        json.dump(metrics_results, f, indent=4)

    