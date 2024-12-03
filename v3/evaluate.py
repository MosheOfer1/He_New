import argparse
import torch
import pandas as pd
import logging

from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianTokenizer, MarianMTModel
from typing import List
from pathlib import Path
import csv
from dataclasses import dataclass
from tqdm import tqdm

from model import CustomLLM


@dataclass
class ModelOutput:
    original_sentence: str
    truncated_sentence: str
    english_translation: str
    llm_completion: str
    final_translation: str
    predicted_word: str
    actual_word: str
    is_correct: bool
    approach_name: str


class TranslationModel:
    def __init__(self, model_name: str, device: str):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)
        self.device = device

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class BaseCompletionApproach:
    def process_sentence(self, sentence: str) -> ModelOutput:
        raise NotImplementedError


class DirectHebrewApproach(BaseCompletionApproach):
    def __init__(self, model_path: str, device: str):
        """
        Initialize the direct Hebrew approach using a model trained on Hebrew text.

        Args:
            model_path: Path to the pretrained model
            tokenizer_path: Path to the tokenizer
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.model.eval()

    def process_sentence(self, sentence: str) -> ModelOutput:
        # Split sentence to get the last word
        sentence = sentence.replace(".", "")
        words = sentence.split()
        truncated_sentence = " ".join(words[:-1])
        actual_word = words[-1]

        # Prepare input for the model
        inputs = self.tokenizer(truncated_sentence, return_tensors="pt").to(self.device)

        # Generate completion
        generation_config = {
            "max_new_tokens": 10,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.encode(" ")[0],  # Use space as EOS token
        }

        outputs = self.model.generate(**inputs, **generation_config)
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the predicted word (everything after the input prompt)
        predicted_text = completion[len(truncated_sentence):].strip()
        predicted_word = predicted_text.split()[0] if predicted_text else ""

        return ModelOutput(
            original_sentence=sentence,
            truncated_sentence=truncated_sentence,
            english_translation="",  # No translation needed
            llm_completion="",  # No English LLM completion
            final_translation=completion,
            predicted_word=predicted_word,
            actual_word=actual_word,
            is_correct=predicted_word == actual_word,
            approach_name="direct_hebrew"
        )


class NaiveApproach(BaseCompletionApproach):
    def __init__(self, he_to_en_model: TranslationModel, en_to_he_model: TranslationModel, llm_model: str, device: str):
        self.he_to_en = he_to_en_model
        self.en_to_he = en_to_he_model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model).to(device)
        self.device = device

    def process_sentence(self, sentence: str) -> ModelOutput:
        # Split sentence and get last word
        words = sentence.split()
        truncated_sentence = " ".join(words[:-1]).replace(".","")
        actual_word = words[-1]

        # Translate to English
        english_translation = self.he_to_en.translate(truncated_sentence).replace(".","")

        # Generate completion with LLM
        prompt = english_translation
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)

        # Configure generation parameters
        generation_config = {
            "max_new_tokens": 10,  # Limit the number of new tokens
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.llm_tokenizer.pad_token_id,
            "eos_token_id": self.llm_tokenizer.encode(" ")[0],  # Use space as EOS token
        }

        outputs = self.llm_model.generate(**inputs, **generation_config)
        completion = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new word (everything after the prompt)
        llm_completion = completion[len(prompt):].split()[0] if completion[len(prompt):].strip() else ""
        llm_completion = prompt + " " + llm_completion

        # Translate back to Hebrew
        final_translation = self.en_to_he.translate(llm_completion).replace(".","")
        predicted_word = final_translation.split()[-1]

        return ModelOutput(
            original_sentence=sentence,
            truncated_sentence=truncated_sentence,
            english_translation=english_translation,
            llm_completion=llm_completion,
            final_translation=final_translation,
            predicted_word=predicted_word,
            actual_word=actual_word,
            is_correct=predicted_word == actual_word,
            approach_name="naive"
        )


class SoftPromptApproach(BaseCompletionApproach):
    def __init__(self, he_to_en_model: TranslationModel,
                 en_to_he_model: TranslationModel, model_path: str, device: str):
        # Load the PEFT configuration
        self.device = device
        self.he_to_en = he_to_en_model
        self.en_to_he = en_to_he_model

        peft_config = PeftConfig.from_pretrained(model_path)

        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)

        # Load the PEFT model
        self.model = PeftModel.from_pretrained(base_model, model_path).to(device)
        self.model.eval()  # Set to evaluation mode

    def process_sentence(self, sentence: str) -> ModelOutput:
        # Split sentence to get the last word
        sentence = sentence.replace(".","")
        words = sentence.split()
        truncated_sentence = " ".join(words[:-1])
        actual_word = words[-1]

        # Translate to English
        english_translation = self.he_to_en.translate(truncated_sentence).replace(".","")

        # Prepare input for soft prompt model
        inputs = self.tokenizer(english_translation, return_tensors="pt").to(self.device)

        # Generate completion with specific parameters for single-word generation
        generation_config = {
            "max_new_tokens": 10,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.encode(" ")[0],  # Use space as EOS token
        }

        outputs = self.model.generate(**inputs, **generation_config)
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the predicted completion
        english_completion = completion[len(english_translation):].strip().split()[0]
        english_full_sentence = f"{english_translation} {english_completion}"

        # Translate back to Hebrew
        final_translation = self.en_to_he.translate(english_full_sentence).replace(".","")
        predicted_word = final_translation.split()[-1]

        return ModelOutput(
            original_sentence=sentence,
            truncated_sentence=truncated_sentence,
            english_translation=english_translation,
            llm_completion=english_full_sentence,
            final_translation=final_translation,
            predicted_word=predicted_word,
            actual_word=actual_word,
            is_correct=predicted_word == actual_word,
            approach_name="soft_prompt"
        )


class CustomModelApproach(BaseCompletionApproach):
    def __init__(
            self,
            he_to_en_model: TranslationModel,
            en_to_he_model: TranslationModel,
            llm_model: str,
            model_path: str,
            device: str,
    ):
        self.device = device

        # Store both translation models
        self.he_to_en_model = he_to_en_model
        self.en_to_he_model = en_to_he_model

        # Load the LLM model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model).to(device)

        # Load the custom model with all components
        self.model = CustomLLM.load_pretrained(
            checkpoint_path=model_path,
            he_en_model=self.he_to_en_model.model,
            en_he_model=self.en_to_he_model.model,
            llm_model=self.llm_model,
            device=device,
            tokenizer3=self.en_to_he_model.tokenizer
        )
        self.model.eval()

    def process_sentence(self, sentence: str) -> ModelOutput:
        # Split sentence to get the last word
        words = sentence.split()
        truncated_sentence = " ".join(words[:-1])
        actual_word = words[-1]

        # Generate completion using the custom model's generate method with all three models
        with torch.no_grad():
            generated_ids = self.model.generate(
                sentence=truncated_sentence,
                he_en_model=self.he_to_en_model.model,
                tokenizer1=self.he_to_en_model.tokenizer,
                tokenizer2=self.llm_tokenizer,
                tokenizer3=self.en_to_he_model.tokenizer,
                device=self.device,
                max_length=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )


        # Decode the generated output
        generated_text = self.en_to_he_model.tokenizer.decode(generated_ids[0], skip_special_tokens=True).replace(".","")
        predicted_word = generated_text.split()[-1]

        return ModelOutput(
            original_sentence=sentence,
            truncated_sentence=truncated_sentence,
            english_translation="",
            llm_completion="",
            final_translation=generated_text,
            predicted_word=predicted_word,
            actual_word=actual_word,
            is_correct=predicted_word == actual_word,
            approach_name="custom_model"
        )


def setup_logging(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/completion.log"),
            logging.StreamHandler()
        ]
    )


def load_sentences(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_results(results: List[ModelOutput], output_file: str):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Approach',
            'Original Sentence',
            'Truncated Sentence',
            'English Translation',
            'LLM Completion',
            'Final Translation',
            'Predicted Word',
            'Actual Word',
            'Is Correct'
        ])

        for result in results:
            writer.writerow([
                result.approach_name,
                result.original_sentence,
                result.truncated_sentence,
                result.english_translation,
                result.llm_completion,
                result.final_translation,
                result.predicted_word,
                result.actual_word,
                result.is_correct
            ])

def parse_args():
    parser = argparse.ArgumentParser(description='NLP Model Evaluator')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the models on')
    parser.add_argument('--he-to-en-model', default='Helsinki-NLP/opus-mt-tc-big-he-en',
                        help='Hebrew to English model name')
    parser.add_argument('--en-to-he-model', default='Helsinki-NLP/opus-mt-en-he',
                        help='English to Hebrew model name')
    parser.add_argument('--llm-model', default='bigscience/bloomz-560m',
                        help='Base LLM model name')
    parser.add_argument('--soft-prompt-model', required=True,
                        help='Path to the soft prompt model')
    parser.add_argument('--custom-model', required=True,
                        help='Path to the custom model checkpoint')
    parser.add_argument('--test-file', required=True,
                        help='Path to the test sentences file')
    parser.add_argument('--log-dir', default='logs',
                        help='Directory for logging')
    parser.add_argument('--output-file', default='model_comparison_results.csv',
                        help='Output file for detailed results')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)

    logger.info("Initializing models...")

    # Initialize translation models
    he_to_en = TranslationModel(args.he_to_en_model, args.device)
    en_to_he = TranslationModel(args.en_to_he_model, args.device)

    # Initialize approaches
    approaches = [
        NaiveApproach(he_to_en, en_to_he, args.llm_model, args.device),
        SoftPromptApproach(he_to_en, en_to_he, args.soft_prompt_model, args.device),
        CustomModelApproach(he_to_en, en_to_he, args.llm_model, args.custom_model, args.device),
        DirectHebrewApproach(args.llm_model, args.device)
    ]

    # Load test sentences
    logger.info(f"Loading test sentences from {args.test_file}")
    sentences = load_sentences(args.test_file)

    # Process sentences with each approach
    all_results = []
    for sentence in tqdm(sentences,):
        for approach in approaches:
            try:
                sentence = sentence.replace(".","")
                result = approach.process_sentence(sentence)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing sentence '{sentence}' with {approach.__class__.__name__}: {str(e)}")

    # Save results
    logger.info(f"Saving results to {args.output_file}")
    save_results(all_results, args.output_file)

    # Calculate and log statistics
    df = pd.DataFrame([vars(r) for r in all_results])
    for approach_name in df['approach_name'].unique():
        approach_results = df[df['approach_name'] == approach_name]
        accuracy = (approach_results['is_correct'].sum() / len(approach_results)) * 100
        logger.info(f"{approach_name.capitalize()} Approach Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()