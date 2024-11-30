import re
import subprocess
import os
import json
from datetime import datetime
import csv



def extract_code_info(project_path):
    try:
       
        symbolsolver_path = "/microDec/symbolsolver"
        
      
        compile_command = ["mvn", "clean", "compile"]
        subprocess.run(compile_command, cwd=symbolsolver_path, check=True, capture_output=True, text=True)
        
        
        package_command = ["mvn", "package", "-DskipTests"]
        subprocess.run(package_command, cwd=symbolsolver_path, check=True, capture_output=True, text=True)

        
        exec_command = ["mvn", "exec:java", "-Dexec.mainClass=Main", f"-Dparse=true", f"-Dproject={project_path}"]
        subprocess.run(exec_command, cwd=symbolsolver_path, check=True, capture_output=True, text=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        if e.stdout:
            print("Standard output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)


def add_app_and_service_to_projects_csv(app_name):
    file_path = "/microDec/data/interfaces/projects.csv" 
    
    row = [app_name, 890, 143]

     
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
         
        writer.writerow(row)

    # check and create service folder for the app.  
    service_path = '/microDec/data/services'
    folder_path = os.path.join(service_path, app_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_interfaces(appName):
    try:
        add_app_and_service_to_projects_csv(appName)
        # Path to the project directory containing the tests
        symbolsolver_path = "/microDec/symbolsolver"
        
        # Run the specific test class
        test_command = ["mvn", "test", "-Dtest=ExtractIdentifiedClassesTest"]
        test_result = subprocess.run(test_command, cwd=symbolsolver_path, check=True, capture_output=True, text=True)
        print("Test output:")
        print(test_result.stdout)
        if test_result.stderr:
            print("Test error output:")
            print(test_result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running tests: {e}")
        if e.stdout:
            print("Standard output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)

#for project file
def generate_id():
    return datetime.now().strftime("%d_%m_%H_%M_%S")

def extract_cluster_data(output_json_path):
    with open(output_json_path, 'r') as file:
        data = json.load(file)
    
    clusters = {}
    for idx, (qualified_name, details) in enumerate(data.items(), start=1):
        clusters[idx] = [details["qualifiedName"]]
    
    return clusters

def extract_project_name(project_path):
    # Split from "src/main/java"
    parts = project_path.split(os.sep)
    return parts[-4] if len(parts) > 4 else parts[-1]

def create_json_structure(project_path, output_json_path, apps_root_path):
    cluster_data = extract_cluster_data(output_json_path)
    
    result = {
        "id": generate_id(),
        "name": extract_project_name(project_path),
        "rootPath": apps_root_path,
        "relativePath": extract_project_name(project_path),
        "clusterString": str(cluster_data),
        "commitHash": ""
    }
    
    return [result]

def save_json(result, output_path):
    with open(output_path, 'w') as file:
        json.dump(result, file, indent=4)           


#END for project file

# Functions for running Java command
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
        
    