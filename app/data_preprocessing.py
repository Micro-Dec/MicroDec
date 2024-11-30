import json
import re
import pandas as pd
from transformers import RobertaTokenizer, BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import csv
def data_preprocessing(output_json_path,appName):
    
    def load_json_file(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    class_data = load_json_file(output_json_path)

    # CodeBERT tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    documents = []
    class_ids = []

    for idx, (class_id, class_info) in enumerate(class_data.items(), start=1):
        tokens = []
        class_ids.append(idx)  #starting from 1
        # Extracttokens
        tokens.append(class_info['qualifiedName'])
        tokens.extend(class_info.get('annotations', []))
        for variable in class_info.get('variables', []):
            tokens.extend(re.findall(r'\w+', variable))
        tokens.extend(class_info.get('dependencies', []))
        tokens.extend(class_info.get('methods', {}).keys())
        for method in class_info.get('methodInvocations', []):
            tokens.append(method['scopeName'])
            tokens.append(method['methodName'])
            tokens.append(method['targetClassName'])
        tokens.extend(class_info.get('implementedTypes', []))
        tokens.extend(class_info.get('extendedTypes', []))
        # all tokens are strings
        tokens = [str(token) for token in tokens]
        documents.append(' '.join(tokens))

    # Tokenize using CodeBERT and create a bag-of-words representation
    tokenized_docs = []
    for doc in documents:
        inputs = tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tokenized_doc = ' '.join(tokens)
        tokenized_docs.append(tokenized_doc)

    # clean tokens
    def custom_tokenizer(text):
        tokens = text.split()
        cleaned_tokens = [token.lstrip('Ġ').lstrip('Ċ').strip() for token in tokens if token.strip() and re.match(r'^\w+$', token)]
        return cleaned_tokens

    #create a bag-of-words model
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, preprocessor=lambda x: x, binary=True, min_df=0.01, max_df=0.90)
    X = vectorizer.fit_transform(tokenized_docs)

    #vocabulary
    vocabulary = vectorizer.get_feature_names_out()

    
    vocabulary_df = pd.DataFrame({
        'token_id': range(1, len(vocabulary) + 1),
        'token': vocabulary
    })

    #vocabulary namesandIDs in the feature vectors
    vocabulary_id_map = {token: token_id for token_id, token in zip(vocabulary_df['token_id'], vocabulary_df['token'])}
    columns = ['class_ID'] + [vocabulary_id_map[token] for token in vocabulary]

    # Feature vectors
    feature_vectors = X.toarray()

    # Prepare data
    csv_data = []
    for i, class_id in enumerate(class_ids):
        row = [class_id] + feature_vectors[i].tolist()
        csv_data.append(row)

    
    df = pd.DataFrame(csv_data, columns=columns)

   
    output_csv_path = f'inputs/output_{appName}.csv'
    df.to_csv(output_csv_path, index=False)

    
    vocabulary_csv_path = f'inputs/vocabulary_{appName}.csv'
    vocabulary_df.to_csv(vocabulary_csv_path, index=False)

    dependencies_file = create_dependencies (output_json_path,appName)

    return output_csv_path, dependencies_file


def create_dependencies (output_json_path,appName):
    def load_json_file(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    
    class_data = load_json_file(output_json_path)

    #mapping from qualifiedName to classid
    class_id_map = {class_info['qualifiedName']: idx for idx, class_info in enumerate(class_data.values(), start=1)}

    # Extract dependencies and create pairs (directed edges)
    edges = []
    for class_id, class_info in enumerate(class_data.values(), start=1):
        dependencies = class_info.get('dependencies', [])
        for dependency in dependencies:
            if dependency in class_id_map:
                dependency_id = class_id_map[dependency]
                edges.append((dependency_id, class_id))  # Cited (dependency) first, citing (current class) second


    output_csv_path = f'inputs/output_dependencies_{appName}.csv'
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['cited', 'citing'])  
        for edge in edges:
            csvwriter.writerow(edge)

    return output_csv_path        




def update_output():
    file_path = '/microDec/app/metrics/output_fosci.csv'
    df = pd.read_csv(file_path, header=None, names=['ClusterID', 'ClassName', 'MethodName', 'MethodParameter', 'MethodReturn'])
    grouped = df.groupby('ClassName')
    filtered_rows = []
    for class_name, group in grouped:
        has_non_empty = group['MethodParameter'].notna().any() or group['MethodReturn'].notna().any() 
        if has_non_empty:
            filtered_group = group[group['MethodParameter'].notna() | group['MethodReturn'].notna()]
        else:
            filtered_group = group
        filtered_rows.append(filtered_group)
    result_df = pd.concat(filtered_rows)
    result_df.to_csv(file_path, index=False, header=False)

