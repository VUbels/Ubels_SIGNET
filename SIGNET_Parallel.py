import pandas as pd
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import argparse
import os
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split

# MLP model definition
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        num = 1
        n = int(input_size)
        len_temp = input_size
        while len_temp > 2.5:
            len_temp = len_temp / 2
            num = num * 2
        num = int(num)
        self.fc1 = nn.Linear(n, num)
        self.fc2 = nn.Linear(num, int(num/4))
        self.fc3 = nn.Linear(int(num/4), 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

# L1 regularization
def l1_regularizer(model, lambda_l1=0.01):
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

# Accuracy calculation
def accuracy(x, y):
    if isinstance(x, torch.Tensor):
        x = x.cpu().data.numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().data.numpy()
    index = 0
    for i in range(y.shape[0]):
        if int(y[i]) == int(x[i]):
            index = index + 1
    return index/y.shape[0]

# Data binarization function
def binarize_data(raw_data):
    binary_data = raw_data.copy()
    threshold_index = raw_data.shape[1] * (raw_data.shape[1] - 1) / 4
    
    for i in range(raw_data.shape[0]):
        gene_expr = raw_data.iloc[i, :]
        marker = [0 for x in range(int(gene_expr.max()) + 1)]
        record = []
        value = []
        
        # Count occurrences of each expression value
        for expr in gene_expr:
            marker[int(expr)] += 1
            
        # Collect non-zero counts
        for j in range(len(marker)):
            if marker[j] != 0:
                value.append(j)
                record.append(marker[j])
                
        # Calculate pairwise averages
        value_calculation = []
        record_calculation = []
        for j in range(len(value)):
            for k in range(j, len(value)):
                value_calculation.append((value[j] + value[k]) / 2)
                if j == k:
                    record_calculation.append(record[j] * (record[j] - 1) / 2)
                else:
                    record_calculation.append(record[j] * record[k])
                    
        # Create dataframe for sorting
        HL_estimator = pd.DataFrame({'index': value_calculation, 'number': record_calculation})
        HL_estimator_new = HL_estimator.sort_values(by=['index'], ascending=True)
        
        # Find threshold and binarize
        thresin = 0
        for j in range(len(record_calculation)):
            thresin += HL_estimator_new.iat[j, 1]
            if thresin >= threshold_index:
                threshold = HL_estimator_new.iat[j, 0]
                binary_data.iloc[i, :] = (binary_data.iloc[i, :] > threshold).astype(int)
                break
                
    return binary_data

# Training function for a single gene
def train_single_gene(gene_idx, data_ntf_binary_train, data_tf_binary_train, opt):
    print(f"Processing gene {gene_idx}")
    iterations = opt.n_epochs
    data_test_train = data_ntf_binary_train[:, [gene_idx]].copy()
    data_X = data_tf_binary_train.copy()
    data_Y = data_test_train.copy()
    
    # Balance dataset if needed
    index = np.sum(data_test_train > 0)
    if index <= np.floor(data_test_train.shape[0] / 6):
        X = list(range(data_test_train.shape[0]))
        pos = int(np.floor(data_test_train.shape[0] / 6)) + 1
        neg = data_test_train.shape[0] - pos
        bootstrapping = []
        pos_flag = neg_flag = 0
        
        while neg_flag < neg or pos_flag < pos:
            sample = int(np.floor(np.random.random() * len(X)))
            if data_test_train[sample] > 0 and pos_flag < pos:
                bootstrapping.append(sample)
                pos_flag += 1
            elif data_test_train[sample] == 0 and neg_flag < neg:
                bootstrapping.append(sample)
                neg_flag += 1
                
        data_X = data_tf_binary_train[bootstrapping, :]
        data_Y = data_test_train[bootstrapping, :]
    
    # Prepare data for training
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.33, random_state=42)
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc = 0
    best_model = None
    
    for attempt in range(10):
        model = MLP(data_tf_binary_train.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
        
        for epoch in range(iterations):
            # Training
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze()) + l1_regularizer(model, 1e-3)
                loss.backward()
                optimizer.step()
            
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.squeeze()).sum().item()
            
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                best_model = model.state_dict()
            
            if best_acc > 0.5:
                break
                
        if best_acc > 0.5:
            break
    
    if best_acc <= 0.5:
        print(f"Gene {gene_idx} failed to achieve >50% accuracy")
        return np.zeros(data_tf_binary_train.shape[1] + 1)
    
    # Load best model and calculate importance scores
    model = MLP(data_tf_binary_train.shape[1]).to(device)
    model.load_state_dict(best_model)
    model.eval()
    
    importance_scores = []
    for i in range(data_tf_binary_train.shape[1]):
        test_data = data_tf_binary_train.copy()
        test_data[:, i] = 0
        test_tensor = torch.from_numpy(test_data).float().to(device)
        with torch.no_grad():
            outputs = model(test_tensor)
            predicted = outputs.max(1)[1]
            acc = accuracy(predicted.cpu(), data_test_train)
            importance_scores.append(acc)
    
    # Add baseline accuracy
    with torch.no_grad():
        baseline = accuracy(
            model(torch.from_numpy(data_tf_binary_train).float().to(device)).max(1)[1].cpu(),
            data_test_train
        )
    importance_scores.append(baseline)
    
    return np.array(importance_scores)

# Argument parser setup
def setup_parser():
    parser = argparse.ArgumentParser(description='SIGNET: Single-cell RNA-seq Gene Regulatory Network')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--n_genes', type=int, default=5000, help='Number of feature genes')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--data_file', type=str, required=True, help='Input scRNA-seq file')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--species', type=str, default="mouse", help='Species (mouse/human)')
    parser.add_argument('--tf_list_file', type=str, required=True, help='TF list file')
    return parser

if __name__ == "__main__":
    opt = parser.parse_args()
    
    # Load and preprocess data
    print("Loading data...")
    data = sc.read_csv(opt.data_file)
    data = data.transpose()
    tf_gene = pd.read_table(opt.tf_list_file, header=None)
    
    # Quality control
    print("Performing quality control...")
    sc.pp.filter_genes(data, min_cells=6)
    mito = 'mt-' if opt.species == "mouse" else 'MT-'
    mito_genes = data.var_names.str.startswith(mito)
    data.var['mito'] = mito_genes
    
    # Calculate QC metrics
    qc = sc.pp.calculate_qc_metrics(data, qc_vars=['mito'])
    cell_qc_dataframe = qc[0]
    data.obs["n_genes"] = cell_qc_dataframe['n_genes_by_counts']
    data.obs["n_counts"] = cell_qc_dataframe['total_counts']
    data.obs["percent_mito"] = cell_qc_dataframe['pct_counts_mito']
    
    # Filter cells
    data = data[data.obs.n_genes < 9000, :]
    data = data[data.obs.percent_mito < 30, :]
    
    # Create raw data matrix
    raw_data = pd.DataFrame(data.X.transpose().copy())
    raw_data.index = data.var_names
    raw_data.columns = data.obs_names
    
    # Normalize and process data
    print("Normalizing data...")
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes=opt.n_genes)
    
    # Get highly variable genes
    gene_list = data.var.highly_variable
    gene_list = gene_list.index[gene_list == True].to_list()
    raw_data_fe = raw_data.loc[gene_list]
    
    # Split into TF and non-TF genes
    list1 = tf_gene[0].to_list()
    list2 = raw_data_fe.index
    list3 = [x for x in list1 if x in list2]  # TF genes
    list4 = [x for x in list2 if x not in list1]  # non-TF genes
    
    raw_data_tf = raw_data.loc[list3, ]
    raw_data_ntf = raw_data_fe.loc[list4, ]
    gene_ntf = raw_data_ntf.index
    gene_tf = raw_data_tf.index
    
    # Binarization process
    print("Starting binarization...")
    data_ntf_binary = binarize_data(raw_data_ntf)
    data_tf_binary = binarize_data(raw_data_tf)
    
    # Save intermediate results
    np.savetxt(f"{opt.output_path}/data_tf_binary.txt", np.asarray(data_tf_binary.transpose()))
    np.savetxt(f"{opt.output_path}/data_ntf_binary.txt", np.asarray(data_ntf_binary.transpose()))
    pd.DataFrame(gene_tf).to_csv(f"{opt.output_path}/gene_tf.csv")
    pd.DataFrame(gene_ntf).to_csv(f"{opt.output_path}/gene_ntf.csv")
    
    # Set up parallel processing
    num_processes = min(cpu_count(), 8)  # Limit to 8 cores max
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs and {num_processes} CPU cores")
    else:
        print(f"Using {num_processes} CPU cores")
    
    # Process non-TF genes
    print("Starting non-TF gene training...")
    data_ntf_binary_train = np.asarray(data_ntf_binary.transpose())
    data_tf_binary_train = np.asarray(data_tf_binary.transpose())
    
    # Create batches for parallel processing
    batch_size = max(1, data_ntf_binary_train.shape[1] // num_processes)
    gene_batches = [
        range(i, min(i + batch_size, data_ntf_binary_train.shape[1]))
        for i in range(0, data_ntf_binary_train.shape[1], batch_size)
    ]
    
    # Initialize result matrix
    coexpressed_result = np.zeros((data_ntf_binary_train.shape[1], 
                                 data_tf_binary_train.shape[1] + 1))
    
    # Process batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        for batch_idx, batch_indices in enumerate(gene_batches):
            print(f"Processing non-TF batch {batch_idx + 1}/{len(gene_batches)}")
            batch_args = [(idx, data_ntf_binary_train, data_tf_binary_train, opt)
                         for idx in batch_indices]
            results = pool.starmap(train_single_gene, batch_args)
            
            for idx, result in zip(batch_indices, results):
                coexpressed_result[idx, :] = result
    
    # Save non-TF results
    np.savetxt(f"{opt.output_path}/co_fc.txt", coexpressed_result)
    
    # Process TF genes
    print("Starting TF gene training...")
    coexpressed_result = np.zeros((data_tf_binary_train.shape[1],
                                 data_tf_binary_train.shape[1] + 1))
    
    # Create batches for TF genes
    batch_size = max(1, data_tf_binary_train.shape[1] // num_processes)
    gene_batches = [
        range(i, min(i + batch_size, data_tf_binary_train.shape[1]))
        for i in range(0, data_tf_binary_train.shape[1], batch_size)
    ]
    
    # Process TF batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        for batch_idx, batch_indices in enumerate(gene_batches):
            print(f"Processing TF batch {batch_idx + 1}/{len(gene_batches)}")
            batch_args = [(idx, data_tf_binary_train, data_tf_binary_train, opt)
                         for idx in batch_indices]
            results = pool.starmap(train_single_gene, batch_args)
            
            for idx, result in zip(batch_indices, results):
                coexpressed_result[idx, :] = result
    
    # Save TF results
    np.savetxt(f"{opt.output_path}/co_tf_fc.txt", coexpressed_result)
    
    print("SIGNET analysis completed!")
    
    # Optional: Generate summary statistics
    tf_network = np.loadtxt(f"{opt.output_path}/co_tf_fc.txt")
    ntf_network = np.loadtxt(f"{opt.output_path}/co_fc.txt")
    
    print("\nNetwork Statistics:")
    print(f"Number of TF-TF interactions: {tf_network.shape[0] * tf_network.shape[1]}")
    print(f"Number of TF-target interactions: {ntf_network.shape[0] * ntf_network.shape[1]}")
    
    # Calculate network density
    tf_density = np.count_nonzero(tf_network > 0.5) / tf_network.size
    ntf_density = np.count_nonzero(ntf_network > 0.5) / ntf_network.size
    print(f"TF-TF network density: {tf_density:.3f}")
    print(f"TF-target network density: {ntf_density:.3f}")