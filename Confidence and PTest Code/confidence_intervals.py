import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib as mpl
import onnxruntime as ort
from tqdm import tqdm

# Function to calculate bootstrap confidence interval
def calculate_bootstrap_ci(words, true_labels, pred_labels, n_bootstrap=200):
    """
    Calculate bootstrap confidence interval for F1 scores at sentence level.
    
    Args:
        words: List of sentences (each sentence is a list of words)
        true_labels: List of true labels (flattened)
        pred_labels: List of predicted labels (flattened)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        tuple: (mean F1 score, standard deviation)
    """
    # Reconstruct sentences for proper bootstrapping
    true_by_sentence = []
    pred_by_sentence = []
    
    index = 0
    for sentence in words:
        sent_true = true_labels[index:index+len(sentence)]
        sent_pred = pred_labels[index:index+len(sentence)]
        
        true_by_sentence.append(sent_true)
        pred_by_sentence.append(sent_pred)
        
        index += len(sentence)
    
    # Bootstrap at sentence level
    f1_scores = []
    unique_labels = list(set(true_labels) | set(pred_labels))  # Union of unique labels
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping", leave=False):
        # Sample sentences with replacement
        indices = np.random.choice(len(words), size=len(words), replace=True)
        
        # Collect labels from sampled sentences
        sample_true = []
        sample_pred = []
        
        for i in indices:
            sample_true.extend(true_by_sentence[i])
            sample_pred.extend(pred_by_sentence[i])
        
        # Calculate F1 score for this sample
        f1 = f1_score(sample_true, sample_pred, average='macro', labels=unique_labels)
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    return mean_f1, std_f1

# Function to create a new folder if it doesn't exist
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# Save classification report with confidence interval
def save_classification_report(report, folder, filename, model_name, mean_f1, std_f1):
    # Convert report to DataFrame
    df = pd.DataFrame(report).transpose()
    
    rows, cols = df.shape
    
    # Create mask for all columns except the last (support)
    mask = np.zeros(df.shape)
    mask[:,cols-1] = True
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Plot heatmap for precision, recall, and f1-score
    ax = sns.heatmap(df, mask=mask, annot=True, cmap=plt.cm.Purples, fmt='.3g',
            vmin=0.0, vmax=1.0,
            linewidths=1, linecolor='black')
    
    # Create mask for all columns except the last (support)
    mask = np.zeros(df.shape)
    mask[:,:cols-1] = True
    
    # Plot heatmap for support column with different color normalization
    ax = sns.heatmap(df, mask=mask, annot=True, cmap=plt.cm.Purples, cbar=False,
            linewidths=1, linecolor='black', fmt='.0f',
            vmin=df['support'].min(),
            vmax=df['support'].sum(),
            norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                      vmax=df['support'].sum()))
    
    # Set title and adjust labels
    title = f"{model_name} - Classification Report\nMacro F1: {mean_f1:.4f} ± {std_f1:.4f}"
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Adjust layout and save figure
    plt.tight_layout(pad=1.1)
    plt.savefig(os.path.join(folder, f'{filename}.png'), dpi=200, bbox_inches='tight')
    plt.close()

# Function to save a confusion matrix as a heatmap
def save_confusion_matrix(y_true, y_pred, labels, folder, filename, model_name, mean_f1, std_f1):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Convert to DataFrame for easier plotting
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot heatmap with logarithmic color scale
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Oranges, fmt='.0f', norm=LogNorm())
    
    # Set title and labels
    title = f"{model_name} - Confusion Matrix\nMacro F1: {mean_f1:.4f} ± {std_f1:.4f}"
    plt.title(title, fontsize=19)
    plt.xlabel('\nPredicted Labels', fontsize=15)
    plt.ylabel('True Labels\n', fontsize=15)
    
    # Adjust layout and save figure
    plt.tight_layout(pad=1.1)
    plt.savefig(os.path.join(folder, f'{filename}.png'), dpi=200, bbox_inches='tight')
    plt.close()

# Function to open and read data
def open_file_get_data_bios(filepath):
    words = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as file:
        word = []
        label = []
        counter = 0
        for line in file:
            split_lines = line.split()
            if len(split_lines) > 0:
                if counter == 128 or (split_lines[0] == "." and counter > 100):
                    if len(split_lines) != 0:
                        if split_lines[0] == ".":
                            word.append(split_lines[0])
                            label.append(split_lines[-1])
                    if len(word) != 0:
                        words.append(word)
                        labels.append(label)
                    word = []
                    label = []
                    counter = 0
                    continue

                word.append(split_lines[0])
                label.append(split_lines[-1])
                counter += 1
    return words, labels

def main():
    print("NER Ensemble Methods Statistical Significance Testing")
    print("======================================================")
    
    # Define label mapping
    global id_to_label
    label_to_tag = {
        'O': 0,
        'B-Drug': 1, 'I-Drug': 2,
        'B-Reason': 3, 'I-Reason': 4,
        'B-Route': 5, 'I-Route': 6,
        'B-Strength': 7, 'I-Strength': 8,
        'B-Form': 9, 'I-Form': 10,
        'B-Dosage': 11, 'I-Dosage': 12,
        'B-Frequency': 13, 'I-Frequency': 14,
        'B-Duration': 15, 'I-Duration': 16,
        'B-ADE': 17, 'I-ADE': 18,
    }
    id_to_label = {i: v for i, v in enumerate(label_to_tag.keys())}
    
    # Load data
    print("Loading data...")
    test_file = "test_spacy.txt"
    words, labels = open_file_get_data_bios(test_file)
    wordsTest = words
    labelsTest = labels
    
    # Load models
    model_checkpoint = [
        "pabRomero/BERT-full-finetuned-ner-pablo",
        "pabRomero/BioBERT-full-finetuned-ner-pablo",
        "pabRomero/ClinicalBERT-full-finetuned-ner-pablo",
        "pabRomero/BioClinicalBERT-full-finetuned-ner-pablo",
        "pabRomero/PubMedBERT-full-finetuned-ner-pablo",
        "pabRomero/BioMedRoBERTa-full-finetuned-ner-pablo",
        "pabRomero/RoBERTa-full-finetuned-ner-pablo",
        "pabRomero/RoBERTa-Large-full-finetuned-ner-pablo"
    ]
    
    model_names = [
        "BERT", "BioBERT", "ClinicalBERT", "BioClinicalBERT", 
        "PubMedBERT", "BioMedRoBERTa", "RoBERTa", "RoBERTa-Large"
    ]
    
    # Create storage for model outputs
    model_predictions = {
        "first_token": [],  # Predictions using first token method
        "max_token": [],    # Predictions using max token method
        "avg_token": []     # Predictions using average token method
    }
    
    model_logits = []  # To store logits for each model
    
    # Create main folder for results
    main_folder = create_folder('./Ensemble_Results')
    
    # Run inference for all models
    print("Running model inference on all models...")
    for idx, model_path in enumerate(tqdm(model_checkpoint, desc="Models")):
        print(f"Processing model: {model_names[idx]}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        # Create dataset
        datasetTest = Dataset.from_dict({"tokens": wordsTest})
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
            return tokenized_inputs
        
        # Tokenize dataset
        tokenized_datasetTest = datasetTest.map(tokenize_and_align_labels, batched=True, remove_columns=datasetTest.column_names)
        
        # Process model predictions with different token strategies
        results = []
        with torch.inference_mode():
            for item in tqdm(tokenized_datasetTest, desc="Processing samples", leave=False):
                input_ids = torch.tensor([item['input_ids']]).to(device)
                attention_mask = torch.tensor([item['attention_mask']]).to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                results.append({
                    'logits': logits.cpu().numpy(),
                    'tokens': tokenizer.convert_ids_to_tokens(item['input_ids'])
                })
        
        # Process predictions using different token strategies
        first_token_labels = []
        max_token_labels = []
        avg_token_labels = []
        model_logits_current = []
        
        resultCounter = 0
        for i, sentences in enumerate(wordsTest):
            resultCounter = 1
            for y, word in enumerate(sentences):
                tokens = ""
                logits = []
                lowerWord = word.lower()
                
                # Process subword tokens
                while tokens != lowerWord:
                    token = results[i]['tokens'][resultCounter].lower()
                    if token[:2] == "##":
                        tokens += token[2:]
                    elif token[0] == "ġ":
                        tokens += token[1:]
                    else:
                        tokens += token
                    logits.append(results[i]['logits'][0][resultCounter])
                    resultCounter += 1
                
                # First token strategy
                first_label = np.argmax(logits[0])
                first_token_labels.append(id_to_label[first_label])
                
                # Max token strategy
                if len(logits) > 1:
                    # Handle BIO tag consistency
                    if first_label != 0 and first_label % 2 != 0:
                        for o, logit in enumerate(logits):
                            if o != 0:
                                maxLogit = np.argmax(logit)
                                if maxLogit % 2 == 0 and maxLogit != 0:
                                    logit[maxLogit - 1] = logit[maxLogit]
                                    logit[maxLogit] = 0
                    
                    max_pos = np.argmax(np.max(logits, axis=1))
                    max_label = np.argmax(logits[max_pos])
                    max_token_labels.append(id_to_label[max_label])
                else:
                    max_token_labels.append(id_to_label[first_label])
                
                # Average token strategy
                if len(logits) > 1:
                    first_pos = np.argmax(logits[0])
                    if first_pos != 0 and first_pos % 2 != 0:
                        for o, logit in enumerate(logits):
                            if o != 0:
                                maxLogit = np.argmax(logit)
                                if maxLogit % 2 == 0:
                                    logit[maxLogit - 1] = logit[maxLogit]
                                    logit[maxLogit] = 0
                    avg_logits = np.mean(logits, axis=0)
                    avg_label = np.argmax(avg_logits)
                    avg_token_labels.append(id_to_label[avg_label])
                else:
                    avg_token_labels.append(id_to_label[first_label])
                
                # Store logits for stacked ensemble
                model_logits_current.append(logits[0])  # Using first token logits
        
        # Store all predictions
        model_predictions["first_token"].append(first_token_labels)
        model_predictions["max_token"].append(max_token_labels)
        model_predictions["avg_token"].append(avg_token_labels)
        model_logits.append(model_logits_current)
    
    # Flatten ground truth labels
    flattened_labels = [item for sublist in labelsTest for item in sublist]
    
    # Implement ensemble methods
    ensemble_results = {}
    
    # 1. Voting First logit Ensemble
    print("\nImplementing Voting First logit Ensemble...")
    voting_first_labels = []
    transposed_labels = np.array(model_predictions["first_token"]).T
    
    for voting_labels in tqdm(transposed_labels, desc="Voting First logit"):
        unique, counts = np.unique(voting_labels, return_counts=True)
        max_index = np.argmax(counts)
        voting_first_labels.append(unique[max_index])
    
    # 2. Voting Max logit Ensemble
    print("\nImplementing Voting Max logit Ensemble...")
    voting_max_labels = []
    transposed_labels = np.array(model_predictions["max_token"]).T
    
    for voting_labels in tqdm(transposed_labels, desc="Voting Max logit"):
        unique, counts = np.unique(voting_labels, return_counts=True)
        max_index = np.argmax(counts)
        voting_max_labels.append(unique[max_index])
    
    # 3. Voting Average Ensemble
    print("\nImplementing Voting Average Ensemble...")
    voting_avg_labels = []
    transposed_labels = np.array(model_predictions["avg_token"]).T
    
    for voting_labels in tqdm(transposed_labels, desc="Voting Average"):
        unique, counts = np.unique(voting_labels, return_counts=True)
        max_index = np.argmax(counts)
        voting_avg_labels.append(unique[max_index])
    
    # 4. Stacked Ensemble first logit
    print("\nImplementing Stacked Ensemble...")
    stacked_labels = []
    try:
        # Load ONNX model
        onnx_model_path = "feedforward_model_stacked.onnx"
        ort_session = ort.InferenceSession(onnx_model_path)
        input_name = ort_session.get_inputs()[0].name
        
        # Run stacked model inference
        for i in tqdm(range(len(model_logits[0])), desc="Stacked Ensemble"):
            # Convert logits to one-hot encoded vectors
            one_hot_input = []
            for y in range(len(model_logits)):
                one_hot = np.zeros(19)  # Assuming 19 classes
                one_hot[np.argmax(model_logits[y][i])] = 1
                one_hot_input.extend(one_hot)
            
            # Prepare input for ONNX model
            onnx_input = np.array(one_hot_input, dtype=np.float32).reshape(1, -1)
            
            # Run ONNX model inference
            ort_inputs = {input_name: onnx_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            onnx_output = ort_outputs[0]
            
            # Get predicted label
            predicted_label_index = np.argmax(onnx_output)
            text_output = id_to_label[predicted_label_index]
            
            stacked_labels.append(text_output)
            
    except Exception as e:
        print(f"Error in stacked ensemble: {e}")
        stacked_labels = voting_first_labels  # Fallback to voting first if stacked fails
    
    # 5. Non-BIO-only-word ensemble
    print("\nImplementing Non-BIO-only-word ensemble...")
    def convert_to_non_bio(labels_list):
        return [label[2:] if label.startswith(('B-', 'I-')) else label for label in labels_list]
    
    non_bio_true_labels = convert_to_non_bio(flattened_labels)
    non_bio_first_labels = convert_to_non_bio(voting_first_labels)
    
    # Store all results
    ensemble_results = {
        "Voting Average Ensemble": voting_avg_labels,
        "Voting First logit Ensemble": voting_first_labels,
        "Voting Max logit Ensemble": voting_max_labels,
        "Stacked Ensemble first logit": stacked_labels,
        "Non-BIO-only-word ensemble": non_bio_first_labels  # Using first token but with BIO tags removed
    }
    
    # True labels for evaluation
    evaluation_labels = {
        "BIO": flattened_labels,
        "Non-BIO": non_bio_true_labels
    }
    
    # Evaluate all ensemble methods
    print("\nEvaluating all ensemble methods...")
    summary_results = []
    
    for method_name, predictions in ensemble_results.items():
        print(f"\nEvaluating {method_name}...")
        
        # Determine if this is a Non-BIO method
        is_non_bio = "Non-BIO" in method_name
        true_labels = evaluation_labels["Non-BIO"] if is_non_bio else evaluation_labels["BIO"]
        
        # Create folder for results
        method_folder = create_folder(os.path.join(main_folder, method_name.replace(" ", "_")))
        
        # Regular evaluation
        reportTable = classification_report(true_labels, predictions, digits=4)
        reportDict = classification_report(true_labels, predictions, output_dict=True)
        
        # Calculate bootstrap confidence interval
        print(f"Calculating bootstrap confidence interval for {method_name}...")
        mean_f1, std_f1 = calculate_bootstrap_ci(wordsTest, true_labels, predictions, n_bootstrap=200)
        
        # Enhanced reporting with confidence interval
        ci_text = f"\nMacro F1 with 95% CI: {mean_f1:.4f} ± {std_f1:.4f}"
        reportTable += ci_text
        
        # Save results
        print(f"Saving results for {method_name}")
        with open(os.path.join(method_folder, f"{method_name.replace(' ', '_')}_report.txt"), 'w') as f:
            f.write(reportTable + "\n")
            f.write(str(reportDict))
        
        # Save visualizations
        save_classification_report(reportDict, method_folder, f"{method_name.replace(' ', '_')}_report", 
                                 method_name, mean_f1, std_f1)
        
        label_set = list(set(true_labels)) if is_non_bio else list(id_to_label.values())
        save_confusion_matrix(true_labels, predictions, label_set, 
                             method_folder, f"{method_name.replace(' ', '_')}_confusion", 
                             method_name, mean_f1, std_f1)
        
        # Store summary results
        summary_results.append({
            "Method": method_name,
            "Macro F1": mean_f1,
            "Std Dev": std_f1,
            "CI": f"{mean_f1:.4f} ± {std_f1:.4f}"
        })
        
        print(f"{method_name} Macro F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Output summary table
    print("\n===== SUMMARY OF RESULTS =====")
    print("Method                      | Macro F1 Score with 95% CI")
    print("-" * 60)
    
    for result in summary_results:
        print(f"{result['Method']:<27} | {result['CI']}")
    
    # Save summary table to file
    with open(os.path.join(main_folder, "summary_results.txt"), 'w') as f:
        f.write("Method,Macro F1,Std Dev,CI\n")
        for result in summary_results:
            f.write(f"{result['Method']},{result['Macro F1']:.4f},{result['Std Dev']:.4f},{result['CI']}\n")
    
    print(f"\nResults saved to {main_folder}")

if __name__ == "__main__":
    main()