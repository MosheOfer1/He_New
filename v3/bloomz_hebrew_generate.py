from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_hebrew_text():
    # Initialize model and tokenizer
    model_name = "bigscience/bloomz-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Hebrew prompt
    prompt = "complete the sentence: I am going to"  # "Complete the sentence: Today I'm going"

    print("Prompt:", prompt)

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(
        inputs.input_ids,
        max_length=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode and print the generated sequences
    print("\nGenerated completions:")

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    generate_hebrew_text()