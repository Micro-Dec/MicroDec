from data_preparation import extract_code_info , get_interfaces, create_json_structure ,save_json,run_java_command
from data_preprocessing import data_preprocessing
from graph import create_graph
from clustering import clustring

import time

if __name__ == "__main__":
    start_time = time.time()  
    
    directory="/microDec"
    appName="production_ssm-master"
      

    project_path = f'/microDec/apps/{appName}/src/main/java'
    type = "bert"
    output_json_path = "/microDec/data/output.json"
    apps_root_path = "/microDec/apps"
    output_path = "projects.json"

    extract_code_info(project_path)
    get_interfaces(appName)
    result = create_json_structure(project_path, output_json_path, apps_root_path)
    save_json(result, output_path)

    run_java_command(directory,project_path)

    nodes_file, dependencies_file = data_preprocessing(output_json_path,appName)

    node_ids, weighted_node_embeddings, num_nodes = create_graph(nodes_file, dependencies_file, type)
    
    num_classes = num_nodes
    clustring_results = clustring(node_ids, weighted_node_embeddings, appName, num_classes)
    end_time = time.time()  
    execution_time = end_time - start_time 
    print(f"Clustring results for app: {clustring_results[0]} AND clustering Method:{clustring_results[1]} AND the metrics reuslts are:{clustring_results[2]} with time : {execution_time:.2f} seconds ")
    print(f"Execution Time: {execution_time:.2f} seconds")




    







