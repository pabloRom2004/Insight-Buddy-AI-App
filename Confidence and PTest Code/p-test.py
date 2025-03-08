import random
import os
import numpy as np
from sklearn.metrics import f1_score
import time
import concurrent.futures

# Base directory
base_dir = "Model_Predictions"

# Function to read lines from a file
def read_labels(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Function to read F1 score from file
def read_f1_score(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            import re
            match = re.search(r'(\d+\.\d+)', content)
            if match:
                return float(match.group(1))
            return float(content)
    except Exception as e:
        print(f"Error reading F1 score from {file_path}: {str(e)}")
        return None

# Function to convert BIO labels to non-BIO
def convert_bio_to_nonbio(bio_labels):
    nonbio_labels = []
    for label in bio_labels:
        if label == 'O':
            nonbio_labels.append('O')
        else:
            # Remove the B- or I- prefix
            parts = label.split('-', 1)
            if len(parts) > 1:
                nonbio_labels.append(parts[1])
            else:
                nonbio_labels.append(label)
    return nonbio_labels

# Define standalone bootstrap iteration function for parallel processing
def bootstrap_iteration(args):
    true_labels, pred_1, pred_2, indices, original_diff = args
    
    # Get bootstrap samples
    boot_true = [true_labels[i] for i in indices]
    boot_pred_1 = [pred_1[i] for i in indices]
    boot_pred_2 = [pred_2[i] for i in indices]
    
    # Calculate F1 scores
    boot_f1_1 = f1_score(boot_true, boot_pred_1, average='macro')
    boot_f1_2 = f1_score(boot_true, boot_pred_2, average='macro')
    boot_diff = boot_f1_2 - boot_f1_1
    
    # Check if the result contradicts the original finding
    return 1 if boot_diff * original_diff <= 0 else 0

# Function to perform bootstrap test with parallel processing
def bootstrap_test(true_labels, pred_1, pred_2, n_iterations=500, is_nonbio=False):
    print(f"Starting bootstrap test with {n_iterations} iterations...")
    start_time = time.time()
    
    # Convert labels if needed for non-BIO format
    if is_nonbio:
        print("Converting ground truth and baseline predictions to non-BIO format...")
        true_labels_adj = convert_bio_to_nonbio(true_labels)
        pred_1_adj = convert_bio_to_nonbio(pred_1)
        # pred_2 is already in non-BIO format
        pred_2_adj = pred_2
    else:
        true_labels_adj = true_labels
        pred_1_adj = pred_1
        pred_2_adj = pred_2
    
    # Calculate original F1 scores using macro average
    original_f1_1 = f1_score(true_labels_adj, pred_1_adj, average='macro')
    original_f1_2 = f1_score(true_labels_adj, pred_2_adj, average='macro')
    original_diff = original_f1_2 - original_f1_1
    
    print(f"Original macro F1 scores: {best_model_name}: {original_f1_1:.4f}, Ensemble: {original_f1_2:.4f}")
    print(f"Original difference: {original_diff:.4f}")
    
    n_samples = len(true_labels)
    
    # Generate bootstrap sample indices
    all_indices = []
    for _ in range(n_iterations):
        indices = [random.randint(0, n_samples-1) for _ in range(n_samples)]
        all_indices.append(indices)
    
    # Prepare arguments for parallel processing
    args_list = [(true_labels_adj, pred_1_adj, pred_2_adj, indices, original_diff) 
                for indices in all_indices]
    
    # Use process-based parallelism
    print("Running bootstrap iterations in parallel...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(bootstrap_iteration, args_list))
    
    p_value = sum(results) / n_iterations
    
    elapsed = time.time() - start_time
    print(f"Bootstrap test completed in {elapsed:.1f} seconds")
    return p_value, original_f1_1, original_f1_2

if __name__ == "__main__":
    # Load ground truth and best single model
    print("Loading data...")
    gt_file = os.path.join(base_dir, "ground_truth_labels.txt")
    ground_truth = read_labels(gt_file)
    
    best_model_file = os.path.join(base_dir, "RoBERTa-Large", "first_token_predictions.txt")
    best_model_name = "RoBERTa-Large (first)"
    best_model_preds = read_labels(best_model_file)
    
    # Calculate macro F1 for best model
    best_macro_f1 = f1_score(ground_truth, best_model_preds, average='macro')
    print(f"Best single model: {best_model_name} with Macro F1: {best_macro_f1:.4f}")
    
    # Test each ensemble method
    ensemble_files = [
        "Non-BIO-only-word_ensemble.txt",
        "Stacked_Ensemble_first_logit.txt",
        "Voting_Average_Ensemble.txt", 
        "Voting_First_logit_Ensemble.txt",
        "Voting_Max_logit_Ensemble.txt"
    ]
    
    results = []
    
    for ens_file in ensemble_files:
        full_path = os.path.join(base_dir, "Ensemble_Results", ens_file)
        f1_file = os.path.join(base_dir, "Ensemble_Results", ens_file.replace(".txt", "_f1.txt"))
        
        if os.path.exists(full_path):
            print(f"\nTesting {ens_file}...")
            ens_preds = read_labels(full_path)
            
            # Check if the lengths match
            if len(ground_truth) != len(ens_preds):
                print(f"ERROR: Length mismatch! Ground truth: {len(ground_truth)}, Ensemble: {len(ens_preds)}")
                continue
            
            # Use pre-calculated F1 score if available
            if os.path.exists(f1_file):
                ens_macro_f1 = read_f1_score(f1_file)
                print(f"Using pre-calculated macro F1 from file: {ens_macro_f1:.4f}")
            else:
                print("No pre-calculated F1 score found")
                if "Non-BIO" in ens_file:
                    # For Non-BIO, convert ground truth to match
                    gt_nonbio = convert_bio_to_nonbio(ground_truth)
                    ens_macro_f1 = f1_score(gt_nonbio, ens_preds, average='macro')
                else:
                    ens_macro_f1 = f1_score(ground_truth, ens_preds, average='macro')
                print(f"Calculated macro F1: {ens_macro_f1:.4f}")
            
            # Check if this is the Non-BIO ensemble
            is_nonbio = "Non-BIO" in ens_file
            if is_nonbio:
                print("Detected Non-BIO format - will convert baseline for comparison")
            
            # Perform bootstrap test
            p_value, base_f1, ens_f1 = bootstrap_test(
                ground_truth, best_model_preds, ens_preds, 
                is_nonbio=is_nonbio
            )
            
            # Format ensemble name for readability
            ens_name = ens_file.replace(".txt", "").replace("_", " ")
            
            # Add result
            results.append({
                "name": ens_name,
                "f1": ens_macro_f1,
                "p_value": p_value,
                "significant": p_value < 0.05
            })
    
    # Print results in a table format
    if results:
        print("\nStatistical Significance Test Results (Macro F1)")
        print("-" * 80)
        print(f"{'Ensemble Method':<30} | {'Macro F1':<10} | {'p-value':<10} | {'Significant':<10}")
        print("-" * 80)
        
        for result in results:
            sig_stars = ""
            if result["p_value"] < 0.001:
                sig_stars = "***"
            elif result["p_value"] < 0.01:
                sig_stars = "**"
            elif result["p_value"] < 0.05:
                sig_stars = "*"
                
            print(f"{result['name']:<30} | {result['f1']:.4f}    | {result['p_value']:.4f}   | {'Yes'+sig_stars if result['significant'] else 'No':<10}")
        
        print("-" * 80)
        print("* p < 0.05, ** p < 0.01, *** p < 0.001")
    else:
        print("\nNo valid results were obtained.")