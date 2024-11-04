import torch
from transformers import MarianMTModel, MarianTokenizer, OPTForCausalLM, AutoTokenizer
import pandas as pd
import logging
from tqdm import tqdm
from model import CustomLLM
from sklearn.metrics import accuracy_score


class PredictionEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.setup_logging()
        self.load_models()
        self.prediction_count = 0

    def setup_logging(self):
        # Set up file handler for detailed logging
        logging.basicConfig(
            filename='model_comparison.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Set up file handler for prediction logging
        self.prediction_logger = logging.getLogger('predictions')
        prediction_handler = logging.FileHandler('predictions.log')
        prediction_handler.setFormatter(logging.Formatter('%(message)s'))
        self.prediction_logger.addHandler(prediction_handler)
        self.prediction_logger.setLevel(logging.INFO)

        # Write header to predictions log
        self.prediction_logger.info("Index\tActual\tNaive Prediction\tCustom Prediction")

    def load_models(self):
        try:
            # Load Helsinki-NLP models
            self.he_to_en_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
            self.en_to_he_model_name = "Helsinki-NLP/opus-mt-en-he"
            self.opt_model_name = "facebook/opt-350m"

            # Load models and tokenizers
            self.he_to_en_tokenizer = MarianTokenizer.from_pretrained(self.he_to_en_model_name)
            self.he_to_en_model = MarianMTModel.from_pretrained(self.he_to_en_model_name).to(self.device)

            self.en_to_he_tokenizer = MarianTokenizer.from_pretrained(self.en_to_he_model_name)
            self.en_to_he_model = MarianMTModel.from_pretrained(self.en_to_he_model_name).to(self.device)

            self.opt_tokenizer = AutoTokenizer.from_pretrained(self.opt_model_name)
            self.opt_model = OPTForCausalLM.from_pretrained(self.opt_model_name).to(self.device)
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def safe_generate(self, model, inputs, **kwargs):
        try:
            return model.generate(**inputs, **kwargs)
        except Exception as e:
            logging.warning(f"Generation failed: {str(e)}")
            return None

    def safe_get_last_word(self, text):
        """Safely extract the last word from text."""
        if not text or not isinstance(text, str):
            return ""
        words = text.strip().split()
        return words[-1] if words else ""

    def naive_approach(self, input_sentence):
        try:
            # Translate Hebrew to English
            inputs = self.he_to_en_tokenizer(input_sentence, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            translated_tokens = self.safe_generate(self.he_to_en_model, inputs)

            if translated_tokens is None:
                return "", "", ""

            english_text = self.he_to_en_tokenizer.decode(translated_tokens[0], skip_special_tokens=True).replace(".",
                                                                                                                  "")

            # Generate with OPT
            opt_inputs = self.opt_tokenizer(english_text, return_tensors="pt", max_length=512, truncation=True)
            opt_inputs = {k: v.to(self.device) for k, v in opt_inputs.items()}
            opt_outputs = self.safe_generate(
                self.opt_model,
                opt_inputs,
                max_new_tokens=1,
                do_sample=True,
                top_p=0.95,
                temperature=0.9
            )

            if opt_outputs is None:
                return "", english_text, ""

            generated_text = self.opt_tokenizer.decode(opt_outputs[0], skip_special_tokens=True)
            predicted_word_en = self.safe_get_last_word(generated_text)

            # Translate back to Hebrew
            if predicted_word_en:
                inputs = self.en_to_he_tokenizer(predicted_word_en, return_tensors="pt", max_length=512,
                                                 truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                translated_tokens = self.safe_generate(self.en_to_he_model, inputs)

                if translated_tokens is None:
                    return "", english_text, predicted_word_en

                predicted_word = self.en_to_he_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            else:
                predicted_word = ""

            return predicted_word, english_text, predicted_word_en
        except Exception as e:
            logging.error(f"Error in naive_approach: {str(e)}")
            return "", "", ""

    def custom_model_approach(self, custom_model, input_sentence):
        try:
            generated_ids = custom_model.generate(
                input_sentence,
                self.he_to_en_model,
                self.he_to_en_tokenizer,
                self.opt_tokenizer,
                self.en_to_he_tokenizer,
                self.device,
                max_length=1
            )
            if generated_ids is None:
                return ""

            generated_text = self.en_to_he_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return self.safe_get_last_word(generated_text)
        except Exception as e:
            logging.error(f"Error in custom_model_approach: {str(e)}")
            return ""

    def evaluate_models(self, custom_model, test_sentences):
        results = []
        error_count = 0

        for sentence in tqdm(test_sentences, desc="Evaluating sentences"):
            try:
                sentence = sentence.strip().replace(".", "")
                if not sentence:
                    continue

                words = sentence.split()
                if len(words) < 2:
                    continue

                input_sentence = " ".join(words[:-1])
                actual_last_word = words[-1]

                # Get predictions from both approaches
                naive_pred, english_text, predicted_word_en = self.naive_approach(input_sentence)
                custom_pred = self.custom_model_approach(custom_model, input_sentence)

                # Log predictions
                self.prediction_count += 1
                self.prediction_logger.info(
                    f"{self.prediction_count}\t{actual_last_word}\t{naive_pred}\t{custom_pred}"
                )

                results.append({
                    "Original_Sentence": sentence,
                    "Input_Sentence": input_sentence,
                    "Actual_Last_Word": actual_last_word,
                    "Naive_Prediction": naive_pred,
                    "Custom_Model_Prediction": custom_pred,
                    "English_Translation": english_text,
                    "English_Predicted_Word": predicted_word_en,
                    "Naive_Correct": naive_pred == actual_last_word,
                    "Custom_Correct": custom_pred == actual_last_word
                })
            except Exception as e:
                error_count += 1
                logging.error(f"Error processing sentence: {str(e)}")
                continue

        if error_count > 0:
            logging.warning(f"Encountered {error_count} errors during evaluation")

        return pd.DataFrame(results)

    def calculate_metrics(self, results_df):
        try:
            metrics = {
                "Naive_Accuracy": accuracy_score(
                    results_df["Actual_Last_Word"] == results_df["Naive_Prediction"],
                    [True] * len(results_df)
                ),
                "Custom_Model_Accuracy": accuracy_score(
                    results_df["Actual_Last_Word"] == results_df["Custom_Model_Prediction"],
                    [True] * len(results_df)
                ),
                "Total_Samples": len(results_df),
                "Naive_Correct_Count": sum(results_df["Naive_Correct"]),
                "Custom_Correct_Count": sum(results_df["Custom_Correct"]),
                "Failed_Predictions": sum(results_df["Naive_Prediction"] == ""),
            }
            return metrics
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return None


def main():
    try:
        # Initialize evaluator
        evaluator = PredictionEvaluator()

        # Load your custom model
        he_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-he-en").to(evaluator.device)
        en_he_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-he").to(evaluator.device)
        llm_model = OPTForCausalLM.from_pretrained("facebook/opt-350m").to(evaluator.device)

        # Load your trained model checkpoint
        custom_model = CustomLLM.load_pretrained(
            "../model_checkpoints/model_epoch_5.pt",
            he_en_model,
            en_he_model,
            llm_model,
            evaluator.device
        )

        # Read test sentences
        with open("../../He_LLM/my_datasets/SVLM_Hebrew_Wikipedia_Corpus.txt", "r", encoding="utf-8") as file:
            test_sentences = file.readlines()

        # Evaluate both approaches
        results_df = evaluator.evaluate_models(custom_model, test_sentences)

        # Calculate metrics
        metrics = evaluator.calculate_metrics(results_df)

        if metrics:
            # Print results
            print("\nEvaluation Results:")
            print(f"Total samples evaluated: {metrics['Total_Samples']}")
            print(f"Naive approach accuracy: {metrics['Naive_Accuracy']:.4f}")
            print(f"Custom model accuracy: {metrics['Custom_Model_Accuracy']:.4f}")
            print(f"Naive approach correct predictions: {metrics['Naive_Correct_Count']}")
            print(f"Custom model correct predictions: {metrics['Custom_Correct_Count']}")
            print(f"Failed predictions: {metrics['Failed_Predictions']}")

            # Save detailed results
            results_df.to_csv("model_comparison_results.csv", index=False, encoding="utf-8")

            # Log metrics
            logging.info(f"Evaluation metrics: {metrics}")
        else:
            print("Failed to calculate metrics")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
