import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_and_plot(data, labels, title, output_file=None):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    reduced_data = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    label_mapping = {
        1: "w = -2", 
        2: "w = 0",  
        3: "w = 0.01",  
        4: "w = -0.02",  
        5: "w = -0.002"   
    }
    
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(reduced_data[idx, 0], reduced_data[idx, 1], 
                    label=label_mapping.get(label, f'Initial w{label}'), 
                    s=15, alpha=0.7)
    
    plt.title(title, fontsize=14)
    plt.xlabel('TSNE Component 1', fontsize=12)
    plt.ylabel('TSNE Component 2', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()

base_folder = "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/updated_weights"
algorithms = ["gd", "sgd", "adam"]

for method in algorithms:
    file_path = os.path.join(base_folder, f"{method}_updated.txt")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    data = []  
    labels = []  

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            line = line.strip().rstrip(',')
            row = np.array(line.split(','), dtype=float)
            if len(row) == 785:
                data.append(row)
                label_index = (i // 100) + 1  
                labels.append(label_index)  
            else:
                print(f"Unexpected row length: {len(row)}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue
    
    if data:
        data = np.array(data)  
        labels = np.array(labels)  

    if data.shape[0] == labels.shape[0]:
        tsne_and_plot(data, labels, f'TSNE Visualization for {method.upper()}', output_file=f"{method}_tsne.png")
    else:
        print(f"Data and labels have mismatched dimensions: {data.shape[0]} vs {labels.shape[0]}")
