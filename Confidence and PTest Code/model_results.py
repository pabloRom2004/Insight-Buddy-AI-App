import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
import numpy as np
import os
import json
from sklearn.metrics import f1_score
from tqdm import tqdm

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

# Function to create a new folder if it doesn't exist
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def main():
    print("NER Ensemble Methods Evaluation")
    print("==============================")
    
    # Define label mapping
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
    main_folder = create_folder('./Model_Predictions')
    
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
        
        # Save individual model predictions
        model_folder = create_folder(os.path.join(main_folder, model_names[idx]))
        
        # Save predictions to files (as text files for simplicity)
        with open(os.path.join(model_folder, 'first_token_predictions.txt'), 'w') as f:
            f.write('\n'.join(first_token_labels))
        
        with open(os.path.join(model_folder, 'max_token_predictions.txt'), 'w') as f:
            f.write('\n'.join(max_token_labels))
        
        with open(os.path.join(model_folder, 'avg_token_predictions.txt'), 'w') as f:
            f.write('\n'.join(avg_token_labels))
    
    # Flatten ground truth labels
    flattened_labels = [item for sublist in labelsTest for item in sublist]
    
    # Save ground truth labels
    with open(os.path.join(main_folder, 'ground_truth_labels.txt'), 'w') as f:
        f.write('\n'.join(flattened_labels))
    
    # Save word structure for bootstrapping
    with open(os.path.join(main_folder, 'word_structure.json'), 'w') as f:
        # Save lengths of each sentence for reconstruction
        sentence_lengths = [len(sentence) for sentence in wordsTest]
        json.dump(sentence_lengths, f)
    
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
    # Try to load stacked model if available
    try:
        import onnxruntime as ort
        
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
    
    non_bio_first_labels = convert_to_non_bio(voting_first_labels)
    
    # Store ensemble results
    ensemble_results = {
        "Voting Average Ensemble": voting_avg_labels,
        "Voting First logit Ensemble": voting_first_labels,
        "Voting Max logit Ensemble": voting_max_labels,
        "Stacked Ensemble first logit": stacked_labels,
        "Non-BIO-only-word ensemble": non_bio_first_labels
    }
    
    # Save ensemble predictions
    ensemble_folder = create_folder(os.path.join(main_folder, 'Ensemble_Results'))
    
    for method_name, predictions in ensemble_results.items():
        with open(os.path.join(ensemble_folder, f"{method_name.replace(' ', '_')}.txt"), 'w') as f:
            f.write('\n'.join(predictions))
        
        # Calculate and save F1 score
        if "Non-BIO" in method_name:
            non_bio_true_labels = convert_to_non_bio(flattened_labels)
            f1 = f1_score(non_bio_true_labels, predictions, average='macro')
        else:
            f1 = f1_score(flattened_labels, predictions, average='macro')
        
        with open(os.path.join(ensemble_folder, f"{method_name.replace(' ', '_')}_f1.txt"), 'w') as f:
            f.write(str(f1))
    
    print(f"\nAll model predictions and ensemble results saved to {main_folder}")

if __name__ == "__main__":
    main()