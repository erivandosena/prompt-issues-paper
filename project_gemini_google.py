import os
import streamlit as st
import google.generativeai as genai
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

    The progress bar updates dynamically until `completion_done` is set to True or reaches 100%.
    """
    with tqdm(
        total=100,
        desc="Waiting conclusion",
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [time: {elapsed}]",
        dynamic_ncols=True,
        leave=True
    ) as pbar:
        while not completion_done and pbar.n < 100:
            time.sleep(0.02)
            pbar.update(1)

def analyze_prompt(messages):
    """
    Send a prompt for analysis to the Gemini API and extract the response.

    Args:
        messages (list): A list of dictionaries containing the system, assistant, and user messages.

    Returns:
        tuple: A tuple containing the model's response content and model name.
    """
    try:
        generation_config = {
                    "temperature": 1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 400,
                    "response_mime_type": "text/plain"
        }

        # Generation moderation parameters
        safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
        ]

        system_message = messages[0]["content"]
        user_message = messages[1]["content"]

        model = genai.GenerativeModel(
            # model_name="gemini-1.5-pro",
            model_name="models/gemini-1.5-flash-latest",
            system_instruction=system_message,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )

        response = model.generate_content(user_message)

        model_name = getattr(model, "model_name", "Unknown LLM")

        return response.text, model_name

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
        content = response_content.strip().replace("**", "").replace("\n", "").replace(';', ' ').replace('"', ' ').replace('-', ' ')

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
        return None,None,None,None

def save_results_csv(data, output_file):
    """
    Save the analysis results to a CSV file. Appends new data if the file exists.

    Args:
        data (list of dict): List of dictionaries containing the analyzed data.
        output_file (str): The path to the output CSV file.
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
    Analyze a given prompt for potential "Prompt Smells" and classify them.

    Args:
        input_text (str): The text of the prompt to analyze.
        api_key (str): The API key for accessing the Gemini API.
        output_file (str): The path to save analysis results.
        dataset_name (str): Optional. The name of the dataset for tracking results.

    Returns:
        list: A list of dictionaries containing the prompt analysis results.
    """
    if not api_key:
        st.error("API key not found!")
        return

    progress_bar()
    smells_data = []

    prompt_template = load_prompt_template(PROMPT_TEMPLATE)

    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Prompt for analytics: {input_text}"}
    ]

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

                smells_data = [{
                    "prompt": input_text.strip(),
                    "classification": classification.strip(),
                    "category": category.strip(),
                    "justification": justification.strip(),
                    "words": len(input_text.split()),
                    "characters": len(input_text),
                    "spaces": input_text.count(" "),
                    "dataset": dataset_name.strip(),
                    "llm": model_name
                }]

                return smells_data
            else:
                st.error("The model did not return a valid response.")
                return None
        except Exception as e:
            st.error(f"An error occurred while processing the prompt: {e}")
            return None
        finally:
            completion_done = True

# Streamlit configuration
st.set_page_config(page_title="Gemini LLM - Prompt Smell", page_icon=":gem:", layout="centered")
st.title("LLM Gemini - Prompt Smell Identification")

# API configuration
api_key = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=api_key)
output_file = "manifold_human_prompts_smells.csv"

# Define the prompt
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE")

# Input field for the prompt
input_text = st.text_area("Enter the prompt for analysis", height=100)
if input_text:
    process_prompt_analysis(input_text, api_key, output_file)
