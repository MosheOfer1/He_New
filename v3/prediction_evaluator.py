import argparse
import logging
import string
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

from model import CustomLLM


class LoggerSetup:
    @staticmethod
    def setup(log_dir: Path) -> Tuple[logging.Logger, logging.Logger]:
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main logger
        logging.basicConfig(
            filename=log_dir / 'model_comparison.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Predictions logger
        pred_logger = logging.getLogger('predictions')
        pred_handler = logging.FileHandler(log_dir / 'predictions.log')
        pred_handler.setFormatter(logging.Formatter('%(message)s'))
        pred_logger.addHandler(pred_handler)
        pred_logger.setLevel(logging.INFO)
        pred_logger.info("Index\tActual\tNaive Prediction\tCustom Prediction\tSoft Prompt Prediction")

        return logging.getLogger(), pred_logger



class PredictionEvaluator:
    def __init__(self, config: Dict):
        self.soft_model = None
        self.soft_tokenizer = None
        self.en_to_he_model = None
        self.en_to_he_tokenizer = None
        self.he_to_en_model = None
        self.he_to_en_tokenizer = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.device = config['device']
        self.prediction_count = 0
        self.logger, self.prediction_logger = LoggerSetup.setup(Path(config['log_dir']))
        self.load_llm_model(config['llm_model'])
        self.load_helsinki_models(config['he_to_en_model'], config['en_to_he_model'])
        self.load_soft_prompt_model(config['llm_model'], config['soft_prompt_model'])

    def load_llm_model(self, model_name: str):
        try:
            self.logger.info(f"Loading LLM model: {model_name}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
            ).to(self.device)
            self.llm_model.eval()  # Ensure model is in evaluation mode
            self.logger.info("Successfully loaded LLM model and tokenizer")
        except Exception as e:
            self.logger.error(f"Error loading LLM model: {str(e)}")

    def load_helsinki_models(self, he_to_en_name: str, en_to_he_name: str):
        self.he_to_en_tokenizer = MarianTokenizer.from_pretrained(he_to_en_name)
        self.he_to_en_model = MarianMTModel.from_pretrained(he_to_en_name).to(self.device)

        self.en_to_he_tokenizer = MarianTokenizer.from_pretrained(en_to_he_name)
        self.en_to_he_model = MarianMTModel.from_pretrained(en_to_he_name).to(self.device)

    def load_soft_prompt_model(self, base_model_name: str, model_path: str):
        self.soft_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.soft_model = PeftModel.from_pretrained(base_model, model_path).to(self.device)
        self.soft_model.eval()

    @staticmethod
    def safe_get_last_word(text: str) -> str:
        """Safely extract the last word from text."""
        if not text or not isinstance(text, str):
            return ""
        words = text.strip().split()
        return words[-1] if words else ""

    def get_top_non_punctuation_token(self, logits: torch.Tensor, tokenizer, k: int = 2) -> str:
        """Get the kth highest probability token that isn't punctuation."""
        try:
            probs = F.softmax(logits, dim=-1)
            top_k = min(k * 3, logits.shape[-1])
            topk_probs, topk_indices = torch.topk(probs, top_k)

            tokens = [tokenizer.decode([idx.item()]) for idx in topk_indices[0]]
            valid_tokens = []

            for token, prob, idx in zip(tokens, topk_probs[0], topk_indices[0]):
                if not all(char in string.punctuation for char in token.strip()):
                    valid_tokens.append((token, prob, idx))
                    if len(valid_tokens) >= k:
                        break

            return valid_tokens[k - 1][0].strip() if len(valid_tokens) >= k else ""
        except Exception as e:
            self.logger.error(f"Error in get_top_non_punctuation_token: {str(e)}")
            return ""

    def naive_approach(self, input_sentence):
        try:
            # Step 1: Translate Hebrew to English
            he_to_en_inputs = self.he_to_en_tokenizer(
                input_sentence,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            he_to_en_inputs = {k: v.to(self.device) for k, v in he_to_en_inputs.items()}

            # Generate English translation
            english_tokens = self.safe_generate(self.he_to_en_model, he_to_en_inputs)
            if english_tokens is None:
                return ""

            english_translation = self.he_to_en_tokenizer.decode(
                english_tokens[0],
                skip_special_tokens=True
            ).replace(".","")

            # Generate with LLM
            llm_inputs = self.llm_tokenizer(english_translation, return_tensors="pt", max_length=512, truncation=True)
            llm_inputs = {k: v.to(self.device) for k, v in llm_inputs.items()}

            # Generate new tokens
            llm_tokens = self.safe_generate(self.llm_model, llm_inputs)
            if llm_tokens is None:
                return ""

            # Decode the generated text and get the last token
            generated_text = self.llm_tokenizer.decode(llm_tokens[0], skip_special_tokens=True)
            predicted_token = self.safe_get_last_word(generated_text)

            # Combine the original English text with the predicted token
            english_translation = f"{english_translation} {predicted_token}"

            # Translate complete English sentence back to Hebrew
            en_to_he_inputs = self.en_to_he_tokenizer(
                english_translation,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            en_to_he_inputs = {k: v.to(self.device) for k, v in en_to_he_inputs.items()}

            # Generate Hebrew translation
            hebrew_tokens = self.safe_generate(self.en_to_he_model, en_to_he_inputs)
            if hebrew_tokens is None:
                return ""

            # Get full Hebrew translation
            hebrew_translation = self.en_to_he_tokenizer.decode(
                hebrew_tokens[0],
                skip_special_tokens=True
            )

            # Extract last word from Hebrew translation
            last_word = self.safe_get_last_word(hebrew_translation)

            return last_word

        except Exception as e:
            logging.error(f"Error in translate_and_get_last_word: {str(e)}")
            return ""

    def custom_model_approach(self, custom_model, input_sentence: str) -> str:
        try:
            # Get logits from custom model
            logits = custom_model(
                input_sentence,
                self.he_to_en_model,
                self.he_to_en_tokenizer,
                self.llm_tokenizer,
                self.en_to_he_tokenizer,
                self.device
            )

            if logits is None:
                return ""

            # Get second-best non-punctuation token
            predicted_text = self.get_top_non_punctuation_token(logits, self.en_to_he_tokenizer, k=2)
            return self.safe_get_last_word(predicted_text)
        except Exception as e:
            self.logger.error(f"Error in custom_model_approach: {str(e)}")
            return ""

    def soft_prompting_approach(self, input_sentence: str) -> str:
        try:
            # Translate Hebrew to English
            inputs = self.he_to_en_tokenizer(input_sentence, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            translated_tokens = self.safe_generate(self.he_to_en_model, inputs)

            if translated_tokens is None:
                return ""

            english_text = self.he_to_en_tokenizer.decode(translated_tokens[0], skip_special_tokens=True).replace(".",
                                                                                                                  "")

            # Use soft-prompted model
            inputs = self.llm_tokenizer(english_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get logits from the soft-prompted model
            with torch.no_grad():
                outputs = self.soft_prompt_model(**inputs)
                logits = outputs.logits[:, -1, :]
                predicted_token = self.get_top_non_punctuation_token(logits, self.llm_tokenizer, k=2)

            # Combine English text with predicted token
            full_english_text = f"{english_text} {predicted_token}"

            # Translate back to Hebrew
            inputs = self.en_to_he_tokenizer(full_english_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            translated_tokens = self.safe_generate(self.en_to_he_model, inputs)

            if translated_tokens is None:
                return ""

            full_hebrew_translation = self.en_to_he_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            predicted_word = self.safe_get_last_word(full_hebrew_translation)

            return predicted_word

        except Exception as e:
            self.logger.error(f"Error in soft_prompting_approach: {str(e)}")
            return ""

    def log_prediction(self, actual: str, naive: str, custom: str, soft_prompt: str):
        self.prediction_count += 1
        self.prediction_logger.info(
            f"{self.prediction_count}\t{actual}\t{naive}\t{custom}\t{soft_prompt}"
        )

    def create_result_entry(self, sentence: str, input_sentence: str, actual_last_word: str,
                            naive_pred: str, custom_pred: str, soft_prompt_pred: str) -> Dict:
        return {
            "Original_Sentence": sentence,
            "Input_Sentence": input_sentence,
            "Actual_Last_Word": actual_last_word,
            "Naive_Prediction": naive_pred,
            "Custom_Model_Prediction": custom_pred,
            "Soft_Prompt_Prediction": soft_prompt_pred,
            "Naive_Correct": naive_pred == actual_last_word,
            "Custom_Correct": custom_pred == actual_last_word,
            "Soft_Prompt_Correct": soft_prompt_pred == actual_last_word
        }

    def safe_generate(self, model, inputs: Dict[str, torch.Tensor], **kwargs) -> Optional[torch.Tensor]:
        try:
            return model.generate(**inputs, **kwargs)
        except Exception as e:
            self.logger.warning(f"Generation failed: {str(e)}")
            return None

    def process_sentence(self, sentence: str) -> Tuple[str, List[str]]:
        sentence = sentence.strip().replace(".", "")
        words = sentence.split()
        if len(words) < 2:
            return "", []
        return " ".join(words[:-1]), words

    def evaluate_models(self, custom_model, test_sentences: List[str]) -> pd.DataFrame:
        results = []
        error_count = 0

        for sentence in tqdm(test_sentences, desc="Evaluating sentences"):
            try:
                input_sentence, words = self.process_sentence(sentence)
                if not words:
                    continue

                actual_last_word = words[-1]
                naive_pred = self.naive_approach(input_sentence)
                custom_pred = self.custom_model_approach(custom_model, input_sentence)
                soft_prompt_pred = self.soft_prompting_approach(input_sentence)

                self.log_prediction(actual_last_word, naive_pred, custom_pred, soft_prompt_pred)
                results.append(self.create_result_entry(
                    sentence, input_sentence, actual_last_word,
                    naive_pred, custom_pred, soft_prompt_pred
                ))

            except Exception as e:
                error_count += 1
                self.logger.error(f"Error processing sentence: {str(e)}")
                continue

        if error_count > 0:
            self.logger.warning(f"Encountered {error_count} errors during evaluation")

        return pd.DataFrame(results)

    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        return {
            "Naive_Accuracy": accuracy_score(
                results_df["Actual_Last_Word"] == results_df["Naive_Prediction"],
                [True] * len(results_df)
            ),
            "Custom_Model_Accuracy": accuracy_score(
                results_df["Actual_Last_Word"] == results_df["Custom_Model_Prediction"],
                [True] * len(results_df)
            ),
            "Soft_Prompt_Accuracy": accuracy_score(
                results_df["Actual_Last_Word"] == results_df["Soft_Prompt_Prediction"],
                [True] * len(results_df)
            ),
            "Total_Samples": len(results_df),
            "Naive_Correct_Count": sum(results_df["Naive_Correct"]),
            "Custom_Correct_Count": sum(results_df["Custom_Correct"]),
            "Soft_Prompt_Correct_Count": sum(results_df["Soft_Prompt_Correct"]),
            "Failed_Predictions": sum(
                (results_df["Naive_Prediction"] == "") &
                (results_df["Custom_Model_Prediction"] == "") &
                (results_df["Soft_Prompt_Prediction"] == "")
            )
        }

    def print_metrics(self, metrics: Dict):
        print("\nEvaluation Results:")
        print("=" * 50)
        print(f"Total samples evaluated: {metrics['Total_Samples']}")

        print("\nAccuracy Metrics:")
        print("-" * 30)
        print(f"Naive approach accuracy:     {metrics['Naive_Accuracy']:.4f}")
        print(f"Custom model accuracy:       {metrics['Custom_Model_Accuracy']:.4f}")
        print(f"Soft prompting accuracy:     {metrics['Soft_Prompt_Accuracy']:.4f}")

        print("\nCorrect Predictions:")
        print("-" * 30)
        print(f"Naive approach:              {metrics['Naive_Correct_Count']:>6} predictions")
        print(f"Custom model:                {metrics['Custom_Correct_Count']:>6} predictions")
        print(f"Soft prompting:              {metrics['Soft_Prompt_Correct_Count']:>6} predictions")


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

    try:
        config = vars(args)
        evaluator = PredictionEvaluator(config)

        # Load custom model
        custom_model = CustomLLM.load_pretrained(
            config['custom_model'],
            evaluator.he_to_en_model,
            evaluator.en_to_he_model,
            evaluator.llm_model,
            config['device'],
            None
        )

        # Read test sentences
        with open(config['test_file'], "r", encoding="utf-8") as file:
            test_sentences = file.readlines()

        # Evaluate models
        results_df = evaluator.evaluate_models(custom_model, test_sentences)
        metrics = evaluator.calculate_metrics(results_df)

        # Print and save results
        evaluator.print_metrics(metrics)
        results_df.to_csv(config['output_file'], index=False, encoding="utf-8")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()