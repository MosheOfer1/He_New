import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Using the same AVAILABLE_MODELS dictionary from the original script
AVAILABLE_MODELS = {
    # Standard Instruct Models
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "32b": "Qwen/Qwen2.5-32B-Instruct",
    "72b": "Qwen/Qwen2.5-72B-Instruct",
    # ... (rest of the models remain the same)
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Reversed Qwen Chat - LLM asks questions')
    parser.add_argument(
        '--model',
        type=str,
        choices=AVAILABLE_MODELS.keys(),
        default="0.5b",
        help='Model version to use'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for question generation (0.0-1.0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum number of tokens in questions'
    )
    return parser.parse_args()


def initialize_model(model_name):
    """Initialize the model and tokenizer."""
    print(f"Initializing Qwen in questioner mode with {model_name} model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def generate_question(model, tokenizer, messages, temperature, max_tokens):
    """Generate a question from the model."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate the question
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )

    # Skip the prompt tokens to get only the generated question
    input_length = model_inputs.input_ids.shape[1]
    generated_tokens = output_ids[0][input_length:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    args = parse_arguments()
    model_name = AVAILABLE_MODELS[args.model]

    # Initialize the model and tokenizer
    model, tokenizer = initialize_model(model_name)

    # Initialize conversation history with reversed roles
    messages = [
        {
            "role": "system",
            "content": "You are an interviewer who asks interesting and engaging questions. "
                       "Ask one question at a time and wait for the human's response. "
                       "Your questions should be thought-provoking and encourage detailed answers. "
                       "Acknowledge the human's responses before asking the next question."
        }
    ]

    print("\nReversed Chat initialized! Commands:")
    print("- Type 'quit' or 'exit' to end the conversation")
    print("- Type 'clear' to start a new conversation")
    print("-" * 50)

    while True:
        try:
            # Generate and print the model's question
            question = generate_question(model, tokenizer, messages, args.temperature, args.max_tokens)
            print(f"\nQwen: {question}")

            # Get user's answer
            user_answer = input("\nYou (as assistant): ").strip()

            # Check for commands
            if user_answer.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            elif user_answer.lower() == 'clear':
                messages = [messages[0]]
                print("\nConversation cleared!")
                continue

            # Add the interaction to conversation history
            messages.append({"role": "assistant", "content": question})
            messages.append({"role": "user", "content": user_answer})

        except KeyboardInterrupt:
            print("\n\nExiting gracefully...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            continue


if __name__ == "__main__":
    main()