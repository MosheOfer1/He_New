import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn.functional as F
import string  # Import the string library for punctuation handling
from create_dataset_and_check_functions import check_using_input_and_label

# Define constants for model and data paths
BASE_MODEL_NAME = "bigscience/bloomz-560m"
MODEL_PATH = "10000_model/"
DATASET_PATH = "testing_4000.csv"
OUTPUT_PATH = "output.csv"
CHECK_FUNCTION = check_using_input_and_label
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ְ

# Function to remove punctuation from input text
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


# Load tokenizer and models, setting up PEFT configuration
def load_model_and_tokenizer(base_model_name, model_path):
    print(f"Using device: {DEVICE}")
    print("Loading base model and tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            return_dict=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.to(DEVICE)
        model.eval()  # Set the model to evaluation mode
        print("Base model and prompt-tuned model successfully loaded.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        exit(1)


# Load dataset from CSV and validate columns
def load_and_validate_dataset(dataset_path, required_columns):
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    if not all(column in df.columns for column in required_columns):
        print(f"Error: Missing columns in the dataset. Expected columns: {required_columns}")
        exit(1)
    return df[required_columns]


# Generate words based on a given input prompt
def generate_word(model, tokenizer, prompt_text, max_tokens=2, max_length=128):
    try:
        prompt_text_clean = remove_punctuation(prompt_text)
        print(f"Tokenizing input (without punctuation): {prompt_text_clean}")

        # Prepare input tensors
        inputs = tokenizer(prompt_text_clean, return_tensors="pt", truncation=True, padding="max_length",
                           max_length=max_length)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        print("Generating output from the model...")

        generated_tokens = []
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]

                # Calculate probabilities and select the token with the highest probability
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_probs, dim=-1).unsqueeze(0)

                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=DEVICE)], dim=-1)

        # Decode the generated tokens into text
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Generated output: {decoded_output}")
        return decoded_output
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return f"ERROR: {str(e)}"


# Process each row, generating output and checking for matches
def process_row(row, model, tokenizer):
    eng_input = row["eng_input"]
    eng_label = row["eng_label"]
    heb_label = row["heb_label"]
    heb_input = row["heb_input"]

    eng_input_clean = remove_punctuation(eng_input)
    print(
        f"Row values: eng_input={eng_input_clean}, eng_label={eng_label}, heb_label={heb_label}, heb_input={heb_input}")

    model_gen = generate_word(model, tokenizer, eng_input_clean)
    is_match, _ = CHECK_FUNCTION(eng_input, model_gen, heb_label)

    return {
        "eng_input": eng_input,
        "eng_label": eng_label,
        "heb_label": heb_label,
        "heb_input": heb_input,
        "model_gen": model_gen,
        "is_match": is_match
    }


# Process the entire dataset and generate output for each row
def process_dataset(df, model, tokenizer):
    results = []
    print(f"Processing {df.shape[0]} rows in the dataset...")
    for index, row in df.iterrows():
        print(f"\nProcessing row {index + 1}/{df.shape[0]}...")
        result = process_row(row, model, tokenizer)
        results.append(result)
        print(f"Generated output: {result['model_gen']}")
    return results


# Save results to CSV file
def save_results_to_csv(results, output_path):
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"Results successfully saved to {output_path}.")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")


# Main function to run the process
def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_NAME, MODEL_PATH)

    # Load and validate dataset
    required_columns = ['eng_input', 'eng_label', 'heb_label', 'heb_input']
    df = load_and_validate_dataset(DATASET_PATH, required_columns)

    # Process dataset
    results = process_dataset(df, model, tokenizer)

    # Save results to CSV
    save_results_to_csv(results, OUTPUT_PATH)


# Run the main function
if __name__ == "__main__":
    main()
