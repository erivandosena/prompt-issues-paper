import os
import re
import streamlit as st
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

results = []
completion_done = False

def progress_bar():
    """
    Display a progress bar to indicate the process status.

    The progress bar updates dynamically until `completion_done` is set to True or reaches the maximum value.
    """
    with tqdm(
        total=100,
        desc="Waiting conclusion",
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [time: {elapsed}]",
        dynamic_ncols=True,
        leave=True
    ) as pbar:
        while not completion_done and pbar.n < 100:
            time.sleep(0.02)  # Update the progress bar
            pbar.update(1)

def analyze_prompt(prompt_input):
    """
    Analyze the given prompt using the OpenAI LLM model.

    Args:
        prompt_input (str): The input prompt to analyze.

    Returns:
        tuple: A tuple containing the model's response content and the model name.
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.4,
            max_tokens=200,
            top_p=0.9,
            stream_usage=False
        )

        response = llm.invoke(prompt_input)
        model_name = getattr(llm, "model_name", "Unknown LLM")

        return response.content, model_name

    except Exception as e:
        st.error(f"Error calling the API: {e}")
        return None, None

def process_response(response_content):
    """
    Process the response content to extract classification, justification, and category.

    Args:
        response_content (str): The response content returned by the LLM.

    Returns:
        tuple: A tuple containing the raw content, classification, justification, and category.
    """
    try:
        content = response_content.strip().replace("**", "").replace("\n", "").replace(';', ' ').replace('"', '')

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
            cats = ["Ambiguity", "Complexity", "Incoherence"]
            for cat in cats:
                if cat.lower() in content.lower():
                    category = cat.capitalize()
                    break
            if category is None or category not in cats:
                raise ValueError(f"Invalid or missing category. In: '{content}'")

        return content, classification, justification, category

    except Exception as e:
        st.error(f"Error Process the response content: {e}")
        return None,None,None,None

def save_results_csv(data, output_file):
    """
    Save analysis results to a CSV file. Appends data if the file exists, otherwise creates a new file.

    Args:
        data (list of dict): List of dictionaries with analyzed data.
        output_file (str): Path to the output CSV file.
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
    Analyze the prompt for potential issues ("Prompt Smells") and classify them.

    Args:
        input_text (str): The prompt text to analyze.
        api_key (str): The API key for accessing the OpenAI LLM.
        output_file (str): The file path for saving analysis results.
        dataset_name (str, optional): Name of the dataset for tracking results. Defaults to "Unknown".

    Returns:
        list: A list of dictionaries with prompt analysis results.
    """
    progress_bar()
    smells_data = []

    if not api_key:
        st.error("API key not found!")
        return

    # Define the prompt
    prompt_template = load_prompt_template("prompt_template.txt")

    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "assistant", "content": "User text is solely for Smell's evaluation, does not require a response."},
        {"role": "user", "content": f"Prompt for analytics: {input_text}"}
    ]

    # # Test: Filter the text associated with 'system'
    # system_content = [item['content'] for item in messages if item['role'] == 'system']
    # for content in system_content:
    #     print(content)

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
st.set_page_config(page_title="GPT LLM - Prompt Smell", page_icon=":100:", layout="centered")
st.title("LLM GPT - Prompt Smell Identification")

# API configuration
api_key = os.environ.get('OPENAI_API_KEY')
output_file = "manifold_human_prompts_smells.csv"

# Input field for the prompt
input_text = st.text_area("Enter the prompt for analysis", height=100)
if input_text:
    process_prompt_analysis(input_text, api_key, output_file)