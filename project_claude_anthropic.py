import os
from anthropic import Anthropic
import streamlit as st
from tqdm import tqdm
import time
import re
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

results = []
completion_done = False

def progress_bar():
    """
    Display a progress bar to indicate the process status.

    The progress bar is updated until `completion_done` is set to True
    or reaches the maximum value of 100.
    """
    with tqdm(
        total=100,
        desc="Waiting conclusion",
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [time: {elapsed}]",
        dynamic_ncols=True,
        leave=True
    ) as pbar:
        while not completion_done and pbar.n < 100:
            time.sleep(0.1)
            pbar.update(1)

def analyze_prompt(messages):
    """
    Analyze the given prompt using the Claude LLM model.

    Args:
        messages (list): A list of dictionaries containing system and user messages.

    Returns:
        tuple: A tuple containing the model's response and the model name.
    """
    system_message = messages[0]["system"]
    user_message = messages[0]["user"]

    try:
        response = api_key.messages.create(
            model="claude-3-5-haiku-latest",
            system=system_message,
            messages=user_message,
            temperature=0.6,
            max_tokens=200,
            top_p=0.9
        )
        model_name = response.to_dict().get("model", "Unknown LLM")
        return response, model_name

    except Exception as e:
        st.error(f"Error calling the API: {e}")
        return None, None

def process_response(response_content):
    """
    Process the response content to extract classification, justification, and category.

    Args:
        response_content (object): The response content object returned by the LLM.

    Returns:
        tuple: A tuple containing raw content, classification, justification, and category.
    """
    try:
        for block in response_content.content:
            if hasattr(block, "type") and block.type == "text":
                content = block.text.strip().replace("**", "").replace("\n", "").replace(';', ' ').replace('"', ' ').replace('-', ' ')

                # Extract Classification
                classification = "No Smell" if "No Smell" in content else "With Smell"

                # Extract Justification
                justification_match = re.search(r"Justification[:\-]?\s*(.*?)(?:\n|$)", content, re.DOTALL)
                if justification_match:
                    justification = justification_match.group(1).strip()
                    if not justification.endswith("."):
                        justification += "."

                # Extract Category
                if classification == "No Smell":
                    category = "Optimal"
                elif classification == "With Smell":
                    cats = ["Ambiguity","Complexity","Contradiction","Grammar",
                            "Incoherence","Incompleteness","Inconsistency",
                            "Inappropriateness","Misguidance","Overloading",
                            "Parsing","Polysemy","Redundancy","Restriction",
                            "Subjectivity","Vagueness"]
                    for cat in cats:
                        if cat.lower() in content.lower():
                            category = cat.capitalize()
                            break
                    if category is None or category not in cats:
                        raise ValueError(f"Invalid or missing category. In: '{content}'")

        return content, classification, justification, category

    except Exception as e:
        st.error(f"Error Process the response content: {e}")
        return None, None, None, None

def save_results_csv(data, output_file):
    """
    Save results to a CSV file. If the file exists, append new rows.
    If it doesn't exist, create the file with a header and add rows.

    Args:
        data (list): List of dictionaries with data to save.
        output_file (str): Path to the CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False, sep=";")
    return None

def load_prompt_template(file_path):
    """
    Loads the content of a text file and returns it as a string.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        str: Content of the file loaded as a string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def process_prompt_analysis(input_text, api_key, output_file, dataset_name="Unknown"):
    """
    Process a given prompt and analyze it for prompt smells.

    Args:
        input_text (str): The prompt text to analyze.
        api_key (str): The API key for accessing the LLM.
        output_file (str): The file path for saving results.
        dataset_name (str, optional): The name of the dataset being processed.

    Returns:
        list: A list of dictionaries containing the analyzed prompt data.
    """

    if not api_key:
        st.error("API key not found!")
        return

    progress_bar()
    smells_data = []

    prompt_template = load_prompt_template(PROMPT_TEMPLATE)

    message = [{"role": "user", "content": f"Prompt for analytics: {input_text}"}]
    messages = [{"system": prompt_template, "user": message}]

    with st.spinner("Processing..."):
        completion_done = False
        try:
            result, model_name = analyze_prompt(messages)

            if result:
                st.success("Analysis completed!")
                raw, classification, justification, category = process_response(result)

                # Display results
                st.markdown(f"**LLM Response:**\n{raw}")
                st.markdown(f"**Prompt:** {input_text}")
                st.markdown(f"**Classification:** {classification}")
                st.markdown(f"**Category:** {category}")
                st.markdown(f"**Justification:** {justification}")

                smells_data.append({
                    "prompt": input_text.strip(),
                    "classification": classification.strip(),
                    "category": category.strip(),
                    "justification": justification.strip(),
                    "words": len(input_text.split()),
                    "characters": len(input_text),
                    "spaces": input_text.count(" "),
                    "dataset": dataset_name.strip(),
                    "llm": model_name
                })

                return smells_data

            else:
                st.error("The model did not return a valid response.")
                return None
        except Exception as e:
            st.error(f"An error occurred while processing the prompt: {e}")
            return None
        finally:
            # Signal the progress bar to stop
            completion_done = True

# Streamlit configuration
st.set_page_config(page_title="LLM - Prompt Smell", page_icon=":robot:", layout="centered")
st.title("LLM Claude - Prompt Smell Identification")

# API configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY")
api_key = Anthropic(api_key=API_KEY)
output_file = "manifold_human_prompts_smells.csv"

# Define the prompt
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE")

# Input field for the prompt
input_text = st.text_area("Enter the prompt for analysis", height=100)
if input_text:
    process_prompt_analysis(input_text, api_key, output_file)
