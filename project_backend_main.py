import os
import pandas as pd
import traceback
from project_gpt_openai import process_prompt_analysis as gpt_process
from project_gemini_google import process_prompt_analysis as gemini_process
from project_ollama_nvidia import process_prompt_analysis as llama_process
from project_claude_anthropic import process_prompt_analysis as claude_process

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# API keys
API_KEYS = {
    "gpt": os.environ.get('OPENAI_API_KEY'),
    "gemini": os.environ.get('GEMINI_API_KEY'),
    "llama": os.environ.get('NVIDIA_API_KEY'),
    "claude": os.environ.get('ANTHROPIC_API_KEY')
}

# Models
MODELS = {
    "gpt": gpt_process,
    "gemini": gemini_process,
    "llama": llama_process,
    "claude": claude_process
}

# Input and output files
dataset_file = os.path.join(BASE_DIR, "manifold_human_prompts_sample_5_percent.csv")
output_file = os.path.join(BASE_DIR, "manifold_human_prompts_smells_sample_5_percent.csv")
log_file = os.path.join(BASE_DIR, "log.txt")

# Load initial dataset
df = pd.read_csv(dataset_file, sep=";")

# Check if the output file already exists and load processed IDs
if os.path.exists(output_file):
    processed_df = pd.read_csv(output_file, sep=";")
    processed_ids = set(processed_df['id'])  # Already processed IDs
else:
    processed_df = pd.DataFrame()
    processed_ids = set()

# Filter prompts not yet processed
prompts = df[~df['id'].isin(processed_ids)][['id', 'prompt', 'dataset']]

def log_and_exit(message):
    """
    Log an error message and exit the system.

    Args:
        message (str): Error message to log.
    """
    try:
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(message + "\n")
    except OSError as e:
        print(f"\u26A0 Failed to write to log file: {e}")
    finally:
        print(f"\u26D4 Exiting... {message}")
        exit(1)

def process_prompt_with_models(input_id, input_prompt, input_dataset, models, api_keys, max_retries=3):
    """
    Process a single prompt using all models and consolidate results.
    Retries 3 times in case of failure before stopping the process.

    Args:
        input_id (int): Prompt ID.
        input_prompt (str): Prompt text.
        input_dataset (str): Dataset name.
        models (dict): Dictionary of models and their processing functions.
        api_keys (dict): Dictionary of API keys for the models.
        max_retries (int): Maximum number of attempts for each model.
        timeout (int): Maximum time (in seconds) to wait for each model response.

    Returns:
        dict | None: Consolidated results for the prompt or None if it fails.
    """

    # Initial results structure
    results = {
        "id": input_id,
        "prompt": input_prompt.strip(),
        "gpt_classification": "",
        "gpt_category": "",
        "gpt_justification": "",
        "gemini_classification": "",
        "gemini_category": "",
        "gemini_justification": "",
        "llama_classification": "",
        "llama_category": "",
        "llama_justification": "",
        "claude_classification": "",
        "claude_category": "",
        "claude_justification": "",
        "words": len(input_prompt.split()),
        "characters": len(input_prompt),
        "spaces": input_prompt.count(" "),
        "dataset": input_dataset.strip()
    }

    # Process with each model
    for model_name, process_function in models.items():
        api_key = api_keys.get(model_name)
        if not api_key:
            log_and_exit(f"Error: API key for {model_name.capitalize()} not found.")

        print(f"\u26A1 Running LLM {model_name.capitalize()}...")
        retries = 0
        success = False

        while retries < max_retries:
            try:
                model_result = process_function(input_prompt, api_key, output_file, input_dataset)

                if model_result is None:
                    raise ValueError(f"Invalid response from LLM {model_name.capitalize()}  \u26D4")

                if isinstance(model_result, list):
                    print(f"LLM {model_name.capitalize()} \033[92m\u2713\033[0m")
                    model_result = model_result[0] if model_result else {}

                if isinstance(model_result, dict):
                    results[f"{model_name}_classification"] = model_result.get("classification", "").strip()
                    results[f"{model_name}_category"] = model_result.get("category", "").strip()
                    results[f"{model_name}_justification"] = model_result.get("justification", "").strip()
                    success = True  # Mark as successful
                    break
                else:
                    raise ValueError(f"Unexpected return format from LLM {model_name.capitalize()}.")

            except TimeoutError as e:
                log_and_exit(f"LLM {model_name.capitalize()}: {e}")
            except Exception as e:
                retries += 1
                print(f"\u26A0 Error with LLM {model_name.capitalize()} (attempt {retries}/{max_retries}): {e}")

        if not success:
            log_and_exit(f"\u274C Failed with LLM {model_name.capitalize()} after {max_retries} attempts.")

    return results

if __name__ == "__main__":
    try:
        consolidated_data = []

        # Process individual prompt
        for _, row in prompts.iterrows():
            input_id = row['id']
            input_prompt = row['prompt']
            input_dataset = row['dataset']

            if isinstance(input_prompt, str) and input_prompt.strip():
                print(f"\nProcessing prompt ID {input_id}: \033[38;5;208m{input_prompt}\033[0m Dataset [{input_dataset}]")

                result = process_prompt_with_models(input_id, input_prompt, input_dataset, MODELS, API_KEYS)

                # Save only successful results
                if result is not None:
                    consolidated_data.append(result)

                    # Save results immediately
                    temp_df = pd.DataFrame([result])
                    try:
                        if os.path.exists(output_file):
                            temp_df.to_csv(output_file, mode='a', index=False, sep=";", header=False)
                        else:
                            temp_df.to_csv(output_file, index=False, sep=";")
                    except OSError as e:
                        log_and_exit(f"Error writing to output file: {e}")
                else:
                    print(f"\033[1;31m Processing prompt ID {input_id} failed. Will not be saved!")
            else:
                print(f"\u26D4 Skipping invalid prompt with ID {input_id}.")

        print(f"\nProcess completed. \U0001F389 Data saved in: {output_file}")
    except KeyboardInterrupt:
        log_and_exit("\U0001F6AB Process interrupted by user (Ctrl+C).")
    except Exception as e:
        log_and_exit(f"\u26A0 Unexpected error: {traceback.format_exc()}")