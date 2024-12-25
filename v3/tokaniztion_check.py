import unicodedata

from transformers import AutoTokenizer
from prettytable import PrettyTable


def analyze_text(text: str) -> PrettyTable:
    table = PrettyTable()
    table.field_names = ["Position", "Character", "Unicode Name", "Unicode (Hex)", "UTF-8 Bytes (Hex)",
                         "UTF-8 Bytes (Binary)"]

    for i, char in enumerate(text):
        # Get character bytes
        char_bytes = char.encode('utf-8')
        # Create hex representation of bytes
        hex_repr = ' '.join([f'{b:02x}' for b in char_bytes])
        # Create binary representation of bytes
        binary_repr = ' '.join([f'{b:08b}' for b in char_bytes])
        # Get Unicode name (with error handling)
        try:
            unicode_name = unicodedata.name(char)
        except:
            unicode_name = "Unknown"

        table.add_row([
            i,
            char,
            unicode_name,
            f'U+{ord(char):04x}',
            hex_repr,
            binary_repr
        ])

    return table


def analyze_tokens(tokenizer, encoded, tokens) -> PrettyTable:
    table = PrettyTable()
    table.field_names = ["Position", "Token", "Token ID", "Bytes (Hex)", "Bytes (Binary)", "UTF-8 Representation"]

    for i, token in enumerate(tokens):
        # Get token bytes
        token_bytes = token.encode('utf-8')
        # Create hex representation
        hex_repr = ' '.join([f'{b:02x}' for b in token_bytes])
        # Create binary representation
        binary_repr = ' '.join([f'{b:08b}' for b in token_bytes])
        # Get UTF-8 char codes
        utf8_repr = 'U+' + ' U+'.join([f'{ord(c):04x}' for c in token])

        table.add_row([
            i,
            token,
            encoded['input_ids'][0][i].item(),
            hex_repr,
            binary_repr,
            utf8_repr
        ])

    return table


def test_hebrew_tokenization():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    hebrew_text = "שלום עולם! זוהי בדיקת טוקניזציה."

    print("Original text:", hebrew_text)

    # Analyze input text
    print("\nInput Text Analysis:")
    input_table = analyze_text(hebrew_text)
    print(input_table)

    # Get encoded tokens
    encoded = tokenizer(hebrew_text, return_tensors="pt", add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

    # Analyze tokens
    print("\nToken Analysis:")
    token_table = analyze_tokens(tokenizer, encoded, tokens)
    print(token_table)

    # Show reconstruction
    decoded_text = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
    print("\nDecoded text:", decoded_text)
    print("Original and decoded text match:", hebrew_text == decoded_text)


if __name__ == "__main__":
    test_hebrew_tokenization()