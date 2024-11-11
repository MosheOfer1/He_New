from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import time


class SimpleTokenStreamer(TextStreamer):
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, delay: float = 0.01):
        super().__init__(tokenizer, skip_prompt)
        self.delay = delay

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("SimpleTokenStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        new_text = self.tokenizer.decode(value, skip_special_tokens=True)
        print(new_text, end='', flush=True)
        time.sleep(self.delay)


def generate_direct(prompt: str, model_name: str = "Qwen/Qwen2.5-0.5B"):
    """Generate text directly without chat template."""
    print(f"Loading model {model_name}...")

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Setup streamer
    streamer = SimpleTokenStreamer(tokenizer, skip_prompt=False, delay=0.01)

    print("\nPrompt + Generation:")
    print("-" * 50)

    # Generate with streaming
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id
        )

    print("\n" + "-" * 50)


if __name__ == "__main__":
    while True:
        selected_prompt = input("Enter a prompt: ")

        # Generate using the 0.5B model (you can change to other sizes)
        generate_direct(selected_prompt)
