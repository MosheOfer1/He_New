from transformers import MarianMTModel, MarianTokenizer, OPTForCausalLM, AutoTokenizer
import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename='translation_process.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load Helsinki-NLP translation models and tokenizers
he_to_en_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
en_to_he_model_name = "Helsinki-NLP/opus-mt-en-he"

he_to_en_tokenizer = MarianTokenizer.from_pretrained(he_to_en_model_name)
he_to_en_model = MarianMTModel.from_pretrained(he_to_en_model_name)

en_to_he_tokenizer = MarianTokenizer.from_pretrained(en_to_he_model_name)
en_to_he_model = MarianMTModel.from_pretrained(en_to_he_model_name)

# Load OPT model and tokenizer
opt_model_name = "facebook/opt-350m"
opt_tokenizer = AutoTokenizer.from_pretrained(opt_model_name)
opt_model = OPTForCausalLM.from_pretrained(opt_model_name)


def translate_he_to_en(hebrew_text):
    # Tokenize and translate Hebrew to English
    inputs = he_to_en_tokenizer(hebrew_text, return_tensors="pt", max_length=512, truncation=True)
    translated_tokens = he_to_en_model.generate(**inputs)
    english_text = he_to_en_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    logging.info(f"Translated Hebrew to English: '{hebrew_text}' -> '{english_text}'")
    return english_text


def generate_with_opt(english_text):
    # Tokenize and generate one word response using OPT
    inputs = opt_tokenizer(english_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = opt_model.generate(**inputs, max_new_tokens=1, do_sample=True, top_p=0.95, temperature=0.9)
    generated_text = opt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"Generated response with OPT for input '{english_text}': '{generated_text}'")
    return generated_text


def main():
    # Read Hebrew sentences from file
    with open("../He_LLM/my_datasets/hebrew_text_for_tests.txt", "r", encoding="utf-8") as file:
        sentences = file.readlines()

    log_data = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Split the sentence to all words except the last one
        words = sentence.split()
        if len(words) < 2:
            continue

        input_sentence = " ".join(words[:-1])
        actual_last_word = words[-1]

        # Step 1: Translate Hebrew to English
        english_text = translate_he_to_en(input_sentence)

        # Step 2: Generate response using OPT
        opt_generated_text = generate_with_opt(english_text)

        # Step 3: Compare the generated word with the actual word
        predicted_word = opt_generated_text.split()[-1] if opt_generated_text else ""

        log_data.append({
            "Original Sentence": sentence,
            "Input Sentence": input_sentence,
            "Actual Last Word": actual_last_word,
            "Predicted Word": predicted_word
        })

        logging.info(
            f"Original: '{sentence}' | Input: '{input_sentence}' | Actual Last Word: '{actual_last_word}' | Predicted Word: '{predicted_word}'")

    # Create a log table with the results
    log_df = pd.DataFrame(log_data)
    print(log_df.to_string(index=False))

    # Save the log to a file
    log_df.to_csv("prediction_log.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
