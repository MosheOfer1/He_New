import unittest
import torch
from transformers import MarianTokenizer, MarianMTModel
import tempfile
import os

from v1.dataset import TextDataset, create_dataloaders


class TestTranslationConsistency(unittest.TestCase):
    model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
    tokenizer = None
    model = None
    device = None
    hebrew_sentences = [
        "שלום עולם",
        "מה שלומך",
        "אני אוהב ללמוד שפות חדשות"
    ]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = MarianTokenizer.from_pretrained(cls.model_name)
        cls.model = MarianMTModel.from_pretrained(cls.model_name)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model.to(cls.device)

        # Create a temporary file with the Hebrew sentences
        cls.temp_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
        for sentence in cls.hebrew_sentences:
            cls.temp_file.write(sentence + '\n')
        cls.temp_file.close()

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary file
        os.unlink(cls.temp_file.name)

    def test_translation_consistency(self):
        dataset = TextDataset(self.temp_file.name, self.tokenizer, eval_split=0, max_length=None)
        train_dataloader, _ = create_dataloaders(dataset, batch_size=len(self.hebrew_sentences))

        dataloader_translations = []
        direct_translations = []

        # Get translations using the dataloader
        for batch_x, _, attention_mask in train_dataloader:
            batch_x, attention_mask = batch_x.to(self.device), attention_mask.to(self.device)
            with torch.no_grad():
                translation = self.model.generate(batch_x, attention_mask=attention_mask)
                decoded_translation = self.tokenizer.batch_decode(translation, skip_special_tokens=True)
                dataloader_translations.extend(decoded_translation)

        # Get direct translations
        for sentence in self.hebrew_sentences:
            direct_input = self.tokenizer(sentence, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                direct_translation = self.tokenizer.decode(
                    self.model.generate(**direct_input)[0],
                    skip_special_tokens=True
                )
                direct_translations.append(direct_translation)

        # Sort both lists of translations
        dataloader_translations.sort()
        direct_translations.sort()

        # Compare sorted lists
        self.assertEqual(dataloader_translations, direct_translations,
                         f"Translations do not match:\nDataloader: {dataloader_translations}\nDirect: {direct_translations}")

        # Print results for inspection
        print("Original sentences:")
        for sentence in self.hebrew_sentences:
            print(f"  {sentence}")
        print("\nDataloader translations:")
        for translation in dataloader_translations:
            print(f"  {translation}")
        print("\nDirect translations:")
        for translation in direct_translations:
            print(f"  {translation}")


if __name__ == "__main__":
    unittest.main()
