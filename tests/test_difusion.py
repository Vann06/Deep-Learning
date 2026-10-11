import math
import unittest

import torch

from src.difusion import (
    NULL_CLASS,
    Classifier,
    UNet,
    ancestral_step,
    confusion_matrix,
    diffusion_loss,
    diversity_per_class,
    fidelity,
    guided_eps,
    linear_schedule,
    q_sample,
    sinusoidal_embedding,
)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.betas, self.alphas, self.alpha_bars = linear_schedule()

    def test_alpha_bars_match_task2(self):
        # valores impresos en la celda 8 de notebooks/02_difusion_forward.ipynb
        expected = {1: 9.999000e-01, 100: 8.970181e-01, 250: 5.240854e-01,
                    500: 7.858724e-02, 750: 3.350550e-03, 1000: 4.035830e-05}
        for t, value in expected.items():
            self.assertTrue(math.isclose(self.alpha_bars[t - 1].item(), value, rel_tol=1e-6), t)

    def test_q_sample_shape_and_t1(self):
        x0 = torch.rand(4, 1, 28, 28) * 2 - 1
        x_t, noise = q_sample(x0, torch.tensor([1, 10, 500, 1000]), self.alpha_bars)
        self.assertEqual(x_t.shape, x0.shape)
        self.assertEqual(noise.shape, x0.shape)
        x_1, _ = q_sample(x0[:1], 1, self.alpha_bars)
        self.assertLess((x_1 - x0[:1]).abs().max().item(), 0.1)


class EmbeddingTests(unittest.TestCase):
    def test_matches_week6_formula(self):
        dim = 16
        t = torch.tensor([1, 7, 500])
        emb = sinusoidal_embedding(t, dim)
        self.assertEqual(emb.shape, (3, dim))
        # el embedding usa float32 (igual que la semana 6); con t grandes el error ronda 1e-5
        for row, pos in enumerate(t.tolist()):
            for i in range(dim // 2):
                angle = pos / 10_000 ** (2 * i / dim)
                self.assertAlmostEqual(emb[row, 2 * i].item(), math.sin(angle), places=4)
                self.assertAlmostEqual(emb[row, 2 * i + 1].item(), math.cos(angle), places=4)


class UNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = UNet()

    def test_output_shape_with_real_and_null_classes(self):
        x = torch.randn(3, 1, 28, 28)
        t = torch.tensor([1, 500, 1000])
        for c in (torch.tensor([0, 5, 9]), torch.full((3,), NULL_CLASS)):
            self.assertEqual(self.model(x, t, c).shape, x.shape)

    def test_loss_is_scalar_and_backpropagates(self):
        _, _, alpha_bars = linear_schedule()
        x0 = torch.rand(4, 1, 28, 28) * 2 - 1
        loss = diffusion_loss(self.model, x0, torch.tensor([0, 1, 2, 3]), alpha_bars, 0.1)
        self.assertEqual(loss.ndim, 0)
        loss.backward()
        self.assertIsNotNone(self.model.out.weight.grad)


class SamplingTests(unittest.TestCase):
    def test_guided_eps_w1_equals_conditional(self):
        torch.manual_seed(0)
        model = UNet().eval()
        x = torch.randn(2, 1, 28, 28)
        t = torch.tensor([300, 700])
        c = torch.tensor([3, 8])
        with torch.no_grad():
            expected = model(x, t, c)
        self.assertTrue(torch.equal(guided_eps(model, x, t, c, 1), expected))

    def test_guided_eps_formula(self):
        class Toy(torch.nn.Module):
            # eps = clase / 10, para poder calcular la guia a mano
            def forward(self, x, t, c):
                return c.float().view(-1, 1, 1, 1) / 10 * torch.ones_like(x)

        x = torch.zeros(1, 1, 28, 28)
        out = guided_eps(Toy(), x, torch.tensor([5]), torch.tensor([4]), 3)
        # 1.0 + 3 (0.4 - 1.0) = -0.8
        self.assertTrue(torch.allclose(out, torch.full_like(x, -0.8)))

    def test_ancestral_step_by_hand(self):
        betas, alphas, alpha_bars = linear_schedule()
        x = torch.full((1, 1, 28, 28), 0.5)
        eps = torch.zeros_like(x)
        z = torch.ones_like(x)
        t = 500
        a, b = alphas[t - 1].item(), betas[t - 1].item()
        expected = 0.5 / math.sqrt(a) + math.sqrt(b)
        out = ancestral_step(x, eps, t, betas, alphas, alpha_bars, z)
        self.assertTrue(torch.allclose(out, torch.full_like(x, expected)))
        # en t = 1 no se agrega ruido
        last = ancestral_step(x, eps, 1, betas, alphas, alpha_bars, z)
        self.assertTrue(torch.allclose(last, torch.full_like(x, 0.5 / math.sqrt(alphas[0].item()))))


class MetricTests(unittest.TestCase):
    def test_diversity_known_values(self):
        images = torch.zeros(20, 1, 28, 28)
        labels = torch.arange(10).repeat_interleave(2)
        # en cada clase, una imagen en cero y la otra en uno: distancia sqrt(784) = 28
        images[1::2] = 1.0
        self.assertTrue(torch.allclose(diversity_per_class(images, labels), torch.full((10,), 28.0)))

    def test_fidelity_and_confusion(self):
        class Fixed(torch.nn.Module):
            # siempre predice la clase 0
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 10)
                logits[:, 0] = 1
                return logits

        images = torch.zeros(4, 1, 28, 28)
        labels = torch.tensor([0, 0, 1, 2])
        self.assertAlmostEqual(fidelity(Fixed(), images, labels), 0.5)
        cm = confusion_matrix(torch.zeros(4, dtype=torch.long), labels)
        self.assertEqual(cm[0, 0].item(), 2)
        self.assertEqual(cm[1, 0].item(), 1)
        self.assertEqual(cm.sum().item(), 4)

    def test_classifier_shape(self):
        self.assertEqual(Classifier()(torch.zeros(5, 1, 28, 28)).shape, (5, 10))


if __name__ == "__main__":
    unittest.main()
