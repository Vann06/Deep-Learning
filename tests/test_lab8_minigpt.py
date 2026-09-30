import unittest

import torch

from lab8_minigpt import (
    MiniGPT,
    build_corpus,
    confidence_moments,
    evaluate_loss,
    generate_text,
    run_task2_experiment,
    sample_greedy,
    sample_top_k,
    sample_top_p,
    summarize_results,
    train_model,
)


class MiniGPTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = MiniGPT(
            vocab_size=8,
            block_size=12,
            d_model=16,
            n_heads=4,
            n_layers=3,
            d_ff=32,
        )

    def test_forward_shapes_and_causal_mask(self):
        indices = torch.randint(0, 8, (2, 10))
        logits, loss, maps = self.model(indices, indices, return_attention=True)
        self.assertEqual(logits.shape, (2, 10, 8))
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(len(maps), 3)
        for weights in maps:
            self.assertEqual(weights.shape, (2, 4, 10, 10))
            future = torch.triu(weights, diagonal=1)
            self.assertEqual(float(future.detach().abs().max().item()), 0.0)

    def test_sampling_contracts(self):
        logits = torch.tensor([9.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        index = sample_greedy(logits)
        self.assertEqual(index, 0)
        self.assertIsInstance(index, int)

        torch.manual_seed(1)
        index = sample_top_k(logits, k=3, tau=0.7)
        self.assertIn(index, {0, 1, 2})
        self.assertIsInstance(index, int)

        index = sample_top_p(logits, p=0.70, tau=0.7)
        self.assertTrue(0 <= index < len(logits))
        self.assertIsInstance(index, int)

    def test_generation_adds_exact_number_of_characters(self):
        alphabet = "abcdefgh"
        stoi = {character: index for index, character in enumerate(alphabet)}
        itos = {index: character for character, index in stoi.items()}
        generated = generate_text(
            self.model,
            "ab",
            15,
            "top-k",
            stoi=stoi,
            itos=itos,
            k=4,
            tau=1.0,
        )
        self.assertEqual(len(generated), 17)
        self.assertTrue(generated.startswith("ab"))

    def test_training_and_full_validation_loss(self):
        text = "".join(chr(code) for code in range(32, 97)) * 4
        corpus = build_corpus(text)
        model = MiniGPT(
            vocab_size=65,
            block_size=12,
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
        )
        history = train_model(
            model,
            corpus,
            epochs=1,
            batch_size=4,
            learning_rate=3e-4,
            device=torch.device("cpu"),
            verbose=False,
        )
        validation_loss = evaluate_loss(
            model,
            corpus.val_data,
            block_size=12,
            batch_size=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(history), 1)
        self.assertGreater(validation_loss, 0.0)
        self.assertAlmostEqual(history[0]["val_loss"], validation_loss, places=6)

    def test_task2_experiment_end_to_end(self):
        alphabet = "abcdefgh"
        stoi = {character: index for index, character in enumerate(alphabet)}
        itos = {index: character for character, index in stoi.items()}

        def generate(prompt, max_new_tokens, strategy, **kwargs):
            return generate_text(
                self.model,
                prompt,
                max_new_tokens,
                strategy,
                stoi=stoi,
                itos=itos,
                **kwargs,
            )

        results = run_task2_experiment(
            generate,
            prompt="ab",
            samples_per_configuration=1,
            max_new_tokens=5,
        )
        self.assertEqual(set(results), set("ABCDE"))
        for samples in results.values():
            self.assertEqual(len(samples), 1)
            self.assertEqual(len(samples[0]["text"]), 7)
            self.assertEqual(len(samples[0]["trace"]), 5)

        summary = summarize_results(results, prompt="ab")
        self.assertEqual(len(summary), 5)
        moments = confidence_moments(results["E"][0], count=2)
        self.assertEqual(len(moments["confident"]), 2)
        self.assertEqual(len(moments["uncertain"]), 2)


if __name__ == "__main__":
    unittest.main()
