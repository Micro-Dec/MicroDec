from data_preparation import extract_code_info , get_interfaces, create_json_structure ,save_json,run_java_command
from data_preprocessing import data_preprocessing
from graph import create_graph
from clustering import clustring
import time
import shutil
from subprocess import call
import csv
import json
import os


def update_csv(all_apps_path, final_resutls_path):
    
    with open(all_apps_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        print(f"Initial number of rows: {len(rows)}")  # Debug: Print initial row count
      
    fieldnames = reader.fieldnames
    print(f"Fieldnames: {fieldnames}")  # Debug: Print fieldnames

    
    file_exists = os.path.exists(final_resutls_path)
    with open(final_resutls_path, mode='a', newline='', buffering=1) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Write the header only 
        if not file_exists or os.stat(final_resutls_path).st_size == 0:
            writer.writeheader()

        for row in rows:
            repo_name = row['name']
            user, repo = repo_name.split('/')
            
            temp_clone_path = f"/microDec/apps/temp_{repo}"
            project_path = f'{temp_clone_path}/src/main/java'
            
            # check if therepo directory exists and contains the required path

            if not os.path.exists(temp_clone_path):
                row['Clustering_Method'] = "not found"
                row['CHM'] = 0
                row['CHD'] = 0
                row['IFN'] = 0
                row['CMQ'] = 0
                row['ccoh'] = 0
                row['ccop'] = 0
                row['SMQ'] = 0
                row['scoh'] = 0
                row['scop'] = 0
                row['SERVICES'] = 0
                row['clustering_names'] = 0
                row['time'] = 0
                writer.writerow(row)
                continue

            if not os.path.exists(project_path):
                row['Clustering_Method'] = "path error"
                row['CHM'] = 0
                row['CHD'] = 0
                row['IFN'] = 0
                row['CMQ'] = 0
                row['ccoh'] = 0
                row['ccop'] = 0
                row['SMQ'] = 0
                row['scoh'] = 0
                row['scop'] = 0
                row['SERVICES'] = 0
                row['clustering_names'] = 0
                row['time'] = 0
                writer.writerow(row)
                continue

            
            
            try:
                actual_repo_name = f'temp_{repo}'    
                

                start_time = time.time()
                directory = "/microDec"
                output_json_path = "/microDec/data/output.json"
                apps_root_path = "/microDec/apps"
                output_path = "projects.json"
                type = "bert"
                
                extract_code_info(project_path)
                get_interfaces(actual_repo_name)
                result = create_json_structure(project_path, output_json_path, apps_root_path)
                save_json(result, output_path)

                run_java_command(directory, project_path)

                nodes_file, dependencies_file = data_preprocessing(output_json_path, actual_repo_name)
                node_ids, weighted_node_embeddings, num_nodes = create_graph(nodes_file, dependencies_file, type, actual_repo_name )

                num_classes = num_nodes
                clustring(node_ids, weighted_node_embeddings, actual_repo_name, num_classes)

                end_time = time.time()
                execution_time = end_time - start_time

                row['time'] = f"{execution_time:.2f}"

                
            except Exception as e:
                print(f"Error processing {repo_name}: {e}")
                # Set all results to 0 in case of an error
                row['Clustering_Method'] = f"Error: {e}"
                row['CHM'] = 0
                row['CHD'] = 0
                row['IFN'] = 0
                row['CMQ'] = 0
                row['ccoh'] = 0
                row['ccop'] = 0
                row['SMQ'] = 0
                row['scoh'] = 0
                row['scop'] = 0
                row['SERVICES'] = 0
                row['clustering_names'] = 0
                row['time'] = 0
            
            
            writer.writerow(row)

if __name__ == "__main__":
  

    # Update the CSV file with new data
    all_apps_path = '/microDec/data/final_results/all_clean_apps_scoh.csv'
    final_resutls_path = '/microDec/data/final_results/final_results.csv'
    update_csv(all_apps_path, final_resutls_path)








