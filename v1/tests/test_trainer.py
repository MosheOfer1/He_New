import unittest
import torch
import tempfile
import shutil
import os
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch

from v1.training import Trainer


class TestTrainer(unittest.TestCase):
    @patch('training.print_model_info')
    def setUp(self, mock_print_model_info):
        self.model = MagicMock()
        self.he_en_model = MagicMock()
        self.tokenizer = MagicMock()
        self.device = torch.device("cpu")

        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 2

        self.temp_log_dir = tempfile.mkdtemp()
        self.temp_save_dir = tempfile.mkdtemp()

        self.trainer = Trainer(self.model, self.he_en_model, self.tokenizer, self.device, self.temp_log_dir,
                               self.temp_save_dir)

    def tearDown(self):
        # Close the logger to release the file handler
        for handler in self.trainer.logger.handlers:
            handler.close()
        self.trainer.logger.handlers.clear()

        # Function to handle permission errors
        def on_rm_error(func, path, exc_info):
            # Check if it's a permission error
            if isinstance(exc_info[1], PermissionError):
                os.chmod(path, 0o777)  # Change file permissions
                func(path)  # Try to remove again
            else:
                raise  # Re-raise the exception if it's not a permission error

        # Remove temporary directories
        shutil.rmtree(self.temp_log_dir, onerror=on_rm_error)
        shutil.rmtree(self.temp_save_dir, onerror=on_rm_error)

    def test_evaluate_full(self):
        batch_size = 2
        seq_length = 5
        vocab_size = 1000

        batch_x = torch.randint(0, vocab_size, (batch_size, seq_length))
        batch_y = torch.randint(0, vocab_size, (batch_size, seq_length))
        he_attention_mask = torch.ones((batch_size, seq_length))

        dataset = TensorDataset(batch_x, batch_y, he_attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        self.he_en_model.generate.return_value = torch.randint(0, vocab_size, (batch_size, seq_length))
        self.model.return_value = torch.rand(batch_size, seq_length, vocab_size)

        result = self.trainer.evaluate_full(dataloader, "test_dataset")

        self.assertIn("dataset", result)
        self.assertEqual(result["dataset"], "test_dataset")
        self.assertIn("loss", result)
        self.assertIn("accuracy", result)
        self.assertIn("perplexity", result)

        self.he_en_model.generate.assert_called_once()
        self.model.assert_called_once()

        model_call_args = self.model.call_args[1]
        self.assertIn("he_attention_mask", model_call_args)
        self.assertIn("en_attention_mask", model_call_args)
        self.assertIn("llm_attention_mask", model_call_args)


if __name__ == '__main__':
    unittest.main()