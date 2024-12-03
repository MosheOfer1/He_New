import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import string

# Load the Helsinki-NLP models for Hebrew to English and English to Hebrew
heb_to_eng_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
eng_to_heb_model_name = 'Helsinki-NLP/opus-mt-en-he'

# Initialize the tokenizers and models
heb_to_eng_tokenizer = MarianTokenizer.from_pretrained(heb_to_eng_model_name)
heb_to_eng_model = MarianMTModel.from_pretrained(heb_to_eng_model_name)

eng_to_heb_tokenizer = MarianTokenizer.from_pretrained(eng_to_heb_model_name)
eng_to_heb_model = MarianMTModel.from_pretrained(eng_to_heb_model_name)


def translate(text, tokenizer, model, num_return_sequences=1, max_length=100):
    """
    Translates the text using the specified tokenizer and model.

    Parameters:
    - text: The input text to be translated.
    - tokenizer: The Marian tokenizer for the translation model.
    - model: The Marian model for translation.
    - num_return_sequences: Number of translation sequences to return.
    - max_length: Maximum length of the translated text.

    Returns:
    - A list of translated texts.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs, num_return_sequences=num_return_sequences, num_beams=3,
                                           max_length=max_length)
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        return translated_texts
    except Exception as e:
        print(f"Translation error: {e}")
        return [""]

def check_using_input_and_label(eng_input, eng_label, heb_label):
    """
    Appends English input and label then translates it back to Hebrew and checks if the last word in the sentence
    it matches the Hebrew label.

    Returns a boolean indicating whether the translation matches and a message.
    """
    eng_sentence = preprocess_text(eng_input) + " " + preprocess_text(eng_label)
    back_to_hebrew = translate(eng_sentence, eng_to_heb_tokenizer, eng_to_heb_model)[0]
    back_to_hebrew_words = back_to_hebrew.split()

    if preprocess_text(back_to_hebrew_words[-1]) == preprocess_text(heb_label):
        return True, "Match found using translating English sentence."

    return False, f"Mismatch found using translating English sentence: translated sentence: {back_to_hebrew}"


def check_using_input_and_label_first_tran_label(eng_input, eng_label, heb_label, max_iterations=5):
    """
    First the label is translated and then checked if it is more than one word in Hebrew then remove last word from the English label
    until one word is obtained in Hebrew.
    Translates the English input and label back to Hebrew and checks if it matches the Hebrew label.

    max_iterations (int): The maximum number of iterations allowed to prevent infinite loop.

    Returns:
        tuple: A boolean indicating whether the translation matches and a message.
    """
    label_back_to_hebrew = translate(eng_label, eng_to_heb_tokenizer, eng_to_heb_model)[0]

    iteration = 0  # Initialize a counter for iterations
    print(f"label_back_to_hebrew: {label_back_to_hebrew}")
    # Keep reducing the label until only one word remains in Hebrew translation or max_iterations reached
    while len(label_back_to_hebrew.split()) > 1 and iteration < max_iterations:
        print(f"iteration: {iteration} label_back_to_hebrew - more then 1 word! - reducing eng label...")
        eng_label = eng_label.split()[:-1]  # Remove the last word from the English label
        eng_label = " ".join(eng_label)
        label_back_to_hebrew = translate(eng_label, eng_to_heb_tokenizer, eng_to_heb_model)[0]
        iteration += 1  # Increment the iteration counter

    print(f"label_back_to_hebrew: {label_back_to_hebrew} 1 word!!")
    # Check if the loop ended because of max_iterations limit
    if iteration >= max_iterations:
        return False, f"Reached max iterations limit ({max_iterations}) without finding a match."

    # Continue with the original logic
    return check_using_input_and_label(eng_input, eng_label, heb_label)

def check_using_eng_label_only(eng_label, heb_label):
    """
    NOT USEFUL!
    Translates only the English label back to Hebrew and compares it with the Hebrew label.

    Returns a boolean indicating whether the translation matches and a message.
    """
    back_to_hebrew_eng_label = translate(eng_label, eng_to_heb_tokenizer, eng_to_heb_model)[0]

    if preprocess_text(back_to_hebrew_eng_label) == preprocess_text(heb_label):
        return True, "Match found using translation of the English label."

    return False, f"Mismatch found using translation of the English label: translated label is: {back_to_hebrew_eng_label}"

def check_using_input_and_label_and_first_heb_word(eng_input, eng_label, heb_label):
    """
    NOT USEFUL!
    Translates the English input and label back to Hebrew and checks if the first word of the translation
    matches the Hebrew label.

    Returns a boolean indicating whether the translation matches and a message.
    """
    # Split eng_label into words if it is a string
    eng_label_words = eng_label.split()

    # Check if the first label is a single letter
    if len(eng_label_words[0]) == 1:
        # If eng_label[0] is a single letter and there's a second word, use the second word
        if len(eng_label_words) > 1:
            eng_label = eng_label_words[1]
        else:
            eng_label = eng_label_words[0]  # If no second word, fallback to first
    else:
        eng_label = eng_label_words[0]  # If first word is not a single letter, use it

    return check_using_input_and_label(eng_input, eng_label, heb_label)


def split_into_sentences(text):
    """
    Splits text into sentences by new lines (one sentence per line).
    """
    return text.strip().splitlines()


def extract_first_five_words(sentence):
    """
    Extracts the first five words from a sentence.
    """
    words = sentence.split()
    return ' '.join(words[:5])


def preprocess_text(text):
    """
    Removes punctuation from the text for better comparison.
    """
    return text.translate(str.maketrans("", "", string.punctuation))



def process_hebrew_translation_dataset(input_file, output_csv_file, log_version, buffer_size=10):
    log_file = f"log_v_{log_version}.txt"

    # Read the input text from the dataset.txt file
    with open(input_file, "r", encoding="utf-8") as infile:
        text = infile.read()

    # Split the input text into sentences (one per line)
    sentences = split_into_sentences(text)
    print(f"Detected {len(sentences)} sentences.")  # Log number of detected sentences

    buffer = []  # Buffer to store results before writing to the CSV file

    # Open log file once for all writing
    with open(log_file, "w", encoding="utf-8") as log_f:
        log_f.write("Translation Process Log\n\n")

        for idx, sentence in enumerate(sentences):
            # Extract the first five words from the sentence
            hebrew_chunk = extract_first_five_words(preprocess_text(sentence))
            words = hebrew_chunk.split()

            # Check if there are at least five words in the chunk
            if len(words) < 5:
                print(f"Skipping sentence {idx + 1} as it has less than 5 words.")
                continue

            # Prepare input and label for Hebrew
            heb_input = " ".join(words[:4])  # First four words
            heb_label = words[-1]  # Last word

            print(f"Processing sentence {idx + 1}/{len(sentences)}: {hebrew_chunk}")

            # Translate the input and label from Hebrew to English
            eng_input_options = translate(heb_input, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1)
            eng_label_options = translate(heb_label, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1)

            eng_input = eng_input_options[0] if eng_input_options else None
            eng_label = eng_label_options[0] if eng_label_options else None

            # Skip sentences if translation fails
            if not eng_input or not eng_label:
                print("Skipping due to translation failure.")
                continue

            # Perform back-translation checks for match/mismatch
            label_is_match, message_1 = check_using_eng_label_only(eng_label, heb_label)
            is_match, message_2 = check_using_input_and_label(eng_input, eng_label, heb_label)

            # Log process for the current sentence
            log_f.write(f"Sentence {idx + 1}/{len(sentences)}:\n")
            log_f.write(f"Original Hebrew: {heb_input} | {heb_label}\n")
            log_f.write(f"English Translation: {eng_input} | {eng_label}\n")
            log_f.write(f"{message_1}\n\n")
            log_f.write(f"{message_2}\n\n")

            # Add the results to the buffer
            buffer.append({
                'eng_input': eng_input,
                'eng_label': eng_label,
                'match_using_label': 1 if label_is_match else 0,
                'match_using_sentence': 1 if is_match else 0,
                'heb_input': heb_input,
                'heb_label': heb_label
            })

            # Write to CSV every buffer_size sentences or at the last sentence
            if (idx + 1) % buffer_size == 0 or (idx + 1) == len(sentences):
                write_mode = 'w' if idx < buffer_size else 'a'
                df = pd.DataFrame(buffer, columns=['eng_input', 'eng_label', 'match_using_label',
                                                   'match_using_sentence', 'heb_input', 'heb_label'])
                df.to_csv(output_csv_file, mode=write_mode, index=False, header=(idx < buffer_size), encoding='utf-8')
                buffer.clear()  # Clear the buffer after writing

    print("Processing complete. Results written to CSV.")


def create_final_dataset(input_file, output_csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Filter rows where the match_using_sentence column equals 1
    filtered_df = df[df['match_using_sentence'] == 1]

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_csv_file, index=False)

    print(f"Filtered dataset saved to '{output_csv_file}'")

# Usage example
# create_final_dataset(input_file="training_data.csv", output_csv_file="final_training_data.csv")


# from sqlalchemy.sql import label
# from transformers import MarianMTModel, MarianTokenizer
# import string
# import pandas as pd
#
#
# # Load the Helsinki-NLP models for Hebrew to English and English to Hebrew
# heb_to_eng_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
# eng_to_heb_model_name = 'Helsinki-NLP/opus-mt-en-he'
#
# heb_to_eng_tokenizer = MarianTokenizer.from_pretrained(heb_to_eng_model_name)
# heb_to_eng_model = MarianMTModel.from_pretrained(heb_to_eng_model_name)
#
# eng_to_heb_tokenizer = MarianTokenizer.from_pretrained(eng_to_heb_model_name)
# eng_to_heb_model = MarianMTModel.from_pretrained(eng_to_heb_model_name)
#
#
# def preprocess_text(text):
#     """Remove punctuation from the text."""
#     return text.translate(str.maketrans("", "", string.punctuation))
#
#
# def translate(text, tokenizer, model, num_return_sequences=1, max_length=100):
#     """Translate text using a tokenizer and model with options for multiple translations, beam search, and error handling."""
#     try:
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         translated_tokens = model.generate(**inputs, num_return_sequences=num_return_sequences, num_beams=3, max_length=max_length)
#         translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
#         return translated_texts
#     except Exception as e:
#         print(f"Translation error: {e}")
#         return [""]
#
#
# def split_into_sentences(text):
#     """Split text into sentences using new lines (one sentence per line)."""
#     return text.strip().splitlines()
#
#
# def extract_first_five_words(sentence):
#     """Extract the first five words from a sentence."""
#     words = sentence.split()
#     return ' '.join(words[:5])
#
#
# def check_using_input_and_label(eng_input, eng_label, heb_label, eng_to_heb_tokenizer, eng_to_heb_model):
#     """Translate the entire English sentence back to Hebrew, then check if the last word."""
#     eng_sentence = eng_input + " " + eng_label
#
#     # Translate the last English sentence back to Hebrew
#     back_to_hebrew = translate(eng_sentence, eng_to_heb_tokenizer, eng_to_heb_model)[0]
#     back_to_hebrew = back_to_hebrew.split()
#     if preprocess_text(back_to_hebrew[-1]) == preprocess_text(heb_label):
#         return True, "Match found using translating english sentence."
#
#     return False, f"Mismatch found using translating eng sentence: translated sentence: {back_to_hebrew}:"
#
#
# def check_using_eng_label_only(eng_label, heb_label, eng_to_heb_tokenizer, eng_to_heb_model):
#     """Translate the English label back to Hebrew, Then comparing the result with the hebrew label."""
#
#     back_to_hebrew_eng_label = translate(eng_label, eng_to_heb_tokenizer, eng_to_heb_model)[0]
#
#     if preprocess_text(back_to_hebrew_eng_label) ==preprocess_text(heb_label):
#         return True, "Match found using Translate the English label."
#
#     return False ,f"Mismatch found using Translate the English label: translated label is: {back_to_hebrew_eng_label} "
#     # If none of the checks pass, return the mis
#
# def process_translation(input_file="dataset.txt", output_csv_file="output.csv", log_version="1"):
#     log_file = f"log_v_{log_version}.txt"
#
#     # Read the input text from the dataset.txt file
#     with open(input_file, "r", encoding="utf-8") as infile:
#         text = infile.read()
#
#     # Split the input text into sentences (one per line)
#     sentences = split_into_sentences(text)
#
#     print(f"Detected {len(sentences)} sentences.")  # Log number of detected sentences
#
#     buffer = []  # Buffer to store results before writing to the CSV file
#     buffer_size = 10  # Number of sentences to process before writing to file
#
#     # Open the CSV file to write headers
#     with open(log_file, "w", encoding="utf-8") as log_f:
#         log_f.write("Translation Process Log\n\n")
#
#     for idx, sentence in enumerate(sentences):
#         # Extract the first five words from the sentence
#         hebrew_chunk = extract_first_five_words(preprocess_text(sentence))
#         words = hebrew_chunk.split()
#         # Get the first four words for 'input'
#         heb_input = " ".join(words[:4])
#
#         # Get the last word for 'label'
#         heb_label = words[-1]
#         print(f"Processing sentence {idx + 1}/{len(sentences)}: {hebrew_chunk}")
#
#         if not hebrew_chunk:
#             print(f"Skipping sentence {idx + 1} as it has less than 5 words.")
#             continue
#
#         # Translate the chunk to English (initial translation)
#         eng_input_options = translate(heb_input, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1)
#         eng_label_options = translate(heb_label, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1)
#         eng_input = eng_input_options[0]
#         eng_label = eng_label_options[0]
#
#         if not eng_input or not eng_label:
#             print("Skipping due to translation failure.")
#             continue
#
#         # Check back-translation for matches or mismatches
#         label_is_match, message_1 = check_using_eng_label_only(
#             eng_label, heb_label, eng_to_heb_tokenizer, eng_to_heb_model
#         )
#
#         is_match, message_2 = check_using_input_and_label(
#             eng_input, eng_label, heb_label, eng_to_heb_tokenizer, eng_to_heb_model
#         )
#
#         # Log the process
#         with open(log_file, "a", encoding="utf-8") as log_f:
#             log_f.write(f"Sentence {idx + 1}/{len(sentences)}:\n")
#             log_f.write(f"Original Hebrew: {heb_input} | {heb_label}\n")
#             log_f.write(f"English Translation: {eng_input} | {eng_label}\n")
#             log_f.write(f"{message_1}\n\n")
#             log_f.write(f"{message_2}\n\n")
#
#         # Add the results to the buffer
#         buffer.append({
#             'eng_input': eng_input,
#             'eng_label': eng_label,
#             'match_using_label': 1 if label_is_match else 0,
#             'match_using_sentence': 1 if is_match else 0,
#             'heb_input': heb_input,
#             'heb_label': heb_label
#         })
#
#         # Every buffer_size sentences, write the buffer to the CSV file and clear the buffer
#         if (idx + 1) % buffer_size == 0 or (idx + 1) == len(sentences):
#             # Create a DataFrame with explicit column names
#             df = pd.DataFrame(buffer, columns=['eng_input', 'eng_label', 'match_using_label', 'match_using_sentence', 'heb_input', 'heb_label'])
#             df.to_csv(output_csv_file, mode='a', index=False, header=not bool(idx), encoding='utf-8')
#             buffer.clear()  # Clear the buffer after writing to the file
#
#     print("Processing complete. Results written to CSV.")
#
# # Example usage for processing a text file
# process_translation(input_file="600_sentences.txt", output_csv_file="output.csv")
#
