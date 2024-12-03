import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from create_dataset_and_check_functions import check_using_input_and_label, preprocess_text
import string

# Global counter for print tracking
print_counter = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the tokenizer and model globally
model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


# Function to print messages with a counter
def print_message(message):
    global print_counter
    print_counter += 1
    print(f"Print {print_counter}: {message}")


# Function to remove punctuation from a given text
def remove_punctuation(text):
    print_message(f"Removing punctuation from text: {text}")
    return text.translate(str.maketrans('', '', string.punctuation))


# Function to generate text based on input
def generate_text(prompt, max_new_tokens=2, do_sample=False):
    """
    Generates text based on the input prompt and returns the decoded output.
    """
    print_message(f"Generating text for prompt: {prompt}")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print_message(f"Generated text: {generated_text}")
    return generated_text


# Function to find the completion part of the generated text
def find_completion_by_length(original, generated):
    """
    Returns the part of the generated text that comes after the original.
    If the generated text does not start with the original, a warning message is returned.
    """
    print_message(f"Finding completion. Original: {original}, Generated: {generated}")
    if generated.startswith(original):
        completion = generated[len(original):].strip()
        print_message(f"Completion found: {completion}")
        return completion
    print_message("The sentences are not the same!")
    return "The sentences are not the same!"


# Function to process a single row and handle the generation and comparison logic
def process_row(idx, eng_input, heb_label, max_attempts=3):
    """
    Processes a single row: generates text, checks if it matches the input, and regenerates if necessary.
    """
    print_message(f"Processing row {idx}")
    attempts = 0

    # Remove punctuation from eng_input
    eng_input_clean = remove_punctuation(eng_input)
    generated_text = generate_text(eng_input_clean)

    while preprocess_text(generated_text) == preprocess_text(eng_input_clean) and attempts < max_attempts:
        print_message(f"Exact match found for row {idx}, generating again... Attempt {attempts + 1}")
        generated_text = generate_text(eng_input_clean)
        attempts += 1

    print_message(f"Generated text for row {idx}: {generated_text}")

    completion_only = find_completion_by_length(eng_input_clean, generated_text)
    print_message(f"Completion only for row {idx}: {completion_only}")

    if completion_only == "":
        completion_only = "no-text-generated"

    # Check if the generated text (completion) matches the Hebrew label
    is_match, _ = check_using_input_and_label(eng_input_clean, completion_only, heb_label)
    print_message(f"Match result for row {idx}: {is_match}")

    return completion_only, is_match


# Modified function to include CSV generation with match checking
def checking_llm_using_eng(df,
                           output_csv="results_basic_model_500_old_code.csv"):
    print_message("Starting LLM check using English input")

    # Create a list to store the output rows
    output_rows = []

    # Iterate over the rows of the DataFrame
    for idx, row in df.iterrows():
        eng_input = row['eng_input']
        eng_label = row['eng_label']
        heb_label = row['heb_label']
        heb_input = row['heb_input']

        # Remove punctuation from eng_input before processing
        eng_input_clean = remove_punctuation(eng_input)
        completion, is_match = process_row(idx, eng_input_clean, heb_label)

        # Save the generated model output (completion) to model_gen and add the match result
        model_gen = completion

        # Append the data to the output_rows list
        output_rows.append({
            "eng_input": eng_input,
            "eng_label": eng_label,
            "heb_label": heb_label,
            "heb_input": heb_input,
            "model_gen": model_gen,
            "is_match": is_match  # Boolean column indicating if the model_gen matches the heb_label
        })

    # Convert the list to a DataFrame and save it to a CSV file
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv, index=False)
    print_message(f"Results saved to {output_csv}")


# Main function to iterate over all rows in the DataFrame and process them
def main():
    # Input and output CSV file paths
    input_csv_file = "testing_4000.csv"

    print_message(f"Loading dataset from {input_csv_file}")

    # Read input CSV file into DataFrame
    try:
        df = pd.read_csv(input_csv_file)
        print_message(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print_message(f"Error loading dataset: {e}")
        return

    # Perform LLM checks using English input
    checking_llm_using_eng(df)
    # Uncomment to perform LLM checks using Hebrew input
    # checking_llm_using_heb(df)
    # checking_model_results(df)


# Execute the main function
if __name__ == "__main__":
    print_message("Starting main execution")
    main()
    print_message("Execution completed")
