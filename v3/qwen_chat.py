import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import sys

AVAILABLE_MODELS = {
    # Standard Instruct Models
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "32b": "Qwen/Qwen2.5-32B-Instruct",
    "72b": "Qwen/Qwen2.5-72B-Instruct",

    # GGUF Quantized Models
    "0.5b-gguf": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "1.5b-gguf": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "3b-gguf": "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "7b-gguf": "Qwen/Qwen2.5-7B-Instruct-GGUF",
    "14b-gguf": "Qwen/Qwen2.5-14B-Instruct-GGUF",
    "32b-gguf": "Qwen/Qwen2.5-32B-Instruct-GGUF",
    "72b-gguf": "Qwen/Qwen2.5-72B-Instruct-GGUF",

    # AWQ Quantized Models
    "0.5b-awq": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
    "1.5b-awq": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "3b-awq": "Qwen/Qwen2.5-3B-Instruct-AWQ",
    "7b-awq": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "14b-awq": "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "32b-awq": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "72b-awq": "Qwen/Qwen2.5-72B-Instruct-AWQ",

    # GPTQ Int4 Quantized Models
    "0.5b-gptq-int4": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
    "1.5b-gptq-int4": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    "3b-gptq-int4": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
    "7b-gptq-int4": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "14b-gptq-int4": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
    "32b-gptq-int4": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    "72b-gptq-int4": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",

    # GPTQ Int8 Quantized Models
    "0.5b-gptq-int8": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
    "1.5b-gptq-int8": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
    "3b-gptq-int8": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
    "7b-gptq-int8": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
    "14b-gptq-int8": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
    "32b-gptq-int8": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
    "72b-gptq-int8": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
}


def print_model_options():
    """Print available model options in a structured format."""
    print("\nAvailable Models:")
    print("\nStandard Models:")
    for k in AVAILABLE_MODELS.keys():
        if not any(suffix in k for suffix in ['-gguf', '-awq', '-gptq']):
            print(f"  - {k}")

    print("\nGGUF Quantized Models:")
    for k in AVAILABLE_MODELS.keys():
        if '-gguf' in k:
            print(f"  - {k}")

    print("\nAWQ Quantized Models:")
    for k in AVAILABLE_MODELS.keys():
        if '-awq' in k:
            print(f"  - {k}")

    print("\nGPTQ Quantized Models:")
    for k in AVAILABLE_MODELS.keys():
        if '-gptq' in k:
            print(f"  - {k}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Qwen Chatbot')
    parser.add_argument(
        '--model',
        type=str,
        choices=AVAILABLE_MODELS.keys(),
        default="0.5b",
        help='Model version to use. Run with --list-models to see all options.'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for response generation (0.0-1.0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum number of tokens in response'
    )
    return parser.parse_args()


def print_system_info():
    """Print system information and available CUDA devices."""
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")


class CustomStreamer(TextStreamer):
    """Custom streamer that adds a small delay between tokens for readability."""

    def __init__(self, tokenizer, skip_prompt=True, delay=0.01):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.delay = delay

    def put(self, value):
        """
        Directly prints new tokens as they're generated.
        """
        # Handle batch size > 1
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("SimpleTokenStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        # Skip if this is the prompt and skip_prompt is True
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Decode and print the new tokens
        new_text = self.tokenizer.decode(value, skip_special_tokens=True)
        print(new_text, end='', flush=True)
        time.sleep(self.delay)


def generate_response_stream(model, tokenizer, messages, temperature, max_tokens):
    """Generate a response from the model using proper streaming."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Create custom streamer
    streamer = CustomStreamer(tokenizer, skip_prompt=True, delay=0)

    # Generate with streaming
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id
    )

    # Skip the prompt tokens to get only the generated response
    input_length = model_inputs.input_ids.shape[1]
    generated_tokens = output_ids[0][input_length:]

    # Return the complete response
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def initialize_model(model_name):
    """Initialize the model and tokenizer."""
    print(f"Initializing Qwen chatbot with {model_name} model (this may take a moment)...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure the tokenizer has necessary tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPossible issues:")
        print("1. Insufficient system resources (RAM/GPU memory)")
        print("2. Internet connection problems")
        print("3. Model not found or access restricted")
        print("\nTry using a smaller model or a quantized version:")
        print("- For low memory: use 0.5b or 1.5b models")
        print("- For better efficiency: try GPTQ or AWQ variants")
        print("- For CPU usage: GGUF variants might work better")
        sys.exit(1)


def main():
    # Parse arguments
    args = parse_arguments()

    # If --list-models flag is used, print models and exit
    if args.list_models:
        print_model_options()
        sys.exit(0)

    model_name = AVAILABLE_MODELS[args.model]

    # Print initial information
    print("\nQwen Chatbot Initializer")
    print("-" * 50)
    print(f"Selected model: {model_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print_system_info()

    # Initialize the model and tokenizer
    model, tokenizer = initialize_model(model_name)

    # Initialize conversation history
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        }
    ]

    print("\nQwen Chatbot initialized! Commands:")
    print("- Type 'quit' or 'exit' to end the conversation")
    print("- Type 'clear' to start a new conversation")
    print("- Type 'info' to see system information")
    print("-" * 50)

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'clear':
                messages = [messages[0]]
                print("\nConversation cleared!")
                continue
            elif user_input.lower() == 'info':
                print_system_info()
                continue

            # Skip empty inputs
            if not user_input:
                continue

            # Add user message to history
            messages.append({"role": "user", "content": user_input})

            # Generate and print response token by token
            print("\nQwen:", end=" ", flush=True)
            response = generate_response_stream(model, tokenizer, messages, args.temperature, args.max_tokens)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\nExiting gracefully...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            continue


if __name__ == "__main__":
    main()
