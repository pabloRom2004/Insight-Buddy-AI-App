# pip install optimum dataclasses typing tqdm transformers onnx onnxruntime

# To run script call this command:
# python runConvert.py --model_id "your/Repo-Here" --quantize --hf_token "your_hf_token_here" --target_repo "your/Repo-Here"

import argparse
import os
import subprocess
import sys
import warnings
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def convert_to_onnx(model_id, output_path, quantize=False):
    print(f"Starting conversion of {model_id} to ONNX...")
    cmd = [sys.executable, "-m", "scripts.convert", "--model_id", model_id]
    if quantize:
        cmd.append("--quantize")
    cmd.extend(["--output_parent_dir", str(output_path)])
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Conversion output:\n{result.stdout}")
        if result.stderr:
            print(f"Conversion warnings/errors:\n{result.stderr}")
        print(f"Successfully converted {model_id} to ONNX{'(quantized)' if quantize else ''}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting model to ONNX: {e}")
        print(f"Error output:\n{e.stderr}")
        raise

def find_files(directory, file_types):
    print(f"Searching for files in {directory}")
    found_files = {file_type: None for file_type in file_types}
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file == 'model_quantized.onnx':
                found_files['onnx_quantized'] = file_path
            elif file == 'model.onnx':
                found_files['onnx'] = file_path
            elif file in file_types:
                found_files[file] = file_path
            if file_path:
                print(f"Found file: {file_path}")
    return found_files

def upload_to_huggingface(local_dir, repo_id, token, quantize=False):
    print(f"Starting upload to Hugging Face repository: {repo_id}")
    api = HfApi()
    
    try:
        # Create the repository if it doesn't exist
        try:
            create_repo(repo_id, token=token, exist_ok=True)
            print(f"Repository {repo_id} created or already exists")
        except Exception as e:
            print(f"Error creating repository: {e}")
            raise

        # Find all necessary files
        file_types = [
            'config.json',
            'quantize_config.json',
            'special_tokens_map.json',
            'tokenizer_config.json',
            'tokenizer.json'
        ]
        found_files = find_files(local_dir, file_types)

        # Determine which ONNX file to upload
        onnx_file = found_files['onnx_quantized'] if quantize and found_files['onnx_quantized'] else found_files['onnx']
        if not onnx_file:
            raise FileNotFoundError("No suitable ONNX file found for upload.")

        # Upload the ONNX file
        print(f"Uploading {onnx_file}...")
        api.upload_file(
            path_or_fileobj=onnx_file,
            path_in_repo=f"onnx/{os.path.basename(onnx_file)}",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

        # Upload other files
        for file_type, file_path in found_files.items():
            if file_path and file_type not in ['onnx', 'onnx_quantized']:
                print(f"Uploading {file_path}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=repo_id,
                    repo_type="model",
                    token=token
                )

        print(f"Successfully uploaded files to {repo_id}")
    except Exception as e:
        print(f"Error during upload process: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert and upload Hugging Face model to ONNX format")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID to convert")
    parser.add_argument("--quantize", action="store_true", help="Quantize the model")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--target_repo", type=str, required=True, help="Target Hugging Face repository name")
    
    args = parser.parse_args()

    output_path = Path("temp_onnx")
    output_path.mkdir(exist_ok=True)
    print(f"Created temporary directory for ONNX files: {output_path}")

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            convert_to_onnx(args.model_id, output_path, args.quantize)
            for warning in w:
                print(f"Warning during conversion: {warning.message}")
        
        upload_to_huggingface(output_path, args.target_repo, args.hf_token, args.quantize)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        print(f"Cleaning up temporary directory: {output_path}")
        import shutil
        shutil.rmtree(output_path, ignore_errors=True)

if __name__ == "__main__":
    main()