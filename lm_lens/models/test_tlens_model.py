# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from lm_lens.models.tlens_model import TransformerLensTransparentLm
from lm_lens.models.transparent_lm import ModelInfo


class TransparentLmTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Picking the smallest model possible so that the test runs faster. It's ok to
        # change this model, but you'll need to update tokenization specifics in some
        # tests.
        cls._lm = TransformerLensTransparentLm(
            model_name="facebook/opt-125m",
            device="cpu",
        )

    def setUp(self):
        self._lm.run(["test", "test 1"])
        self._eps = 1e-5

    def test_model_info(self):
        info = self._lm.model_info()
        self.assertEqual(
            info,
            ModelInfo(
                name="facebook/opt-125m",
                n_params_estimate=84934656,
                n_layers=12,
                n_heads=12,
                d_model=768,
                d_vocab=50272,
            ),
        )

    def test_tokens(self):
        tokens = self._lm.tokens()

        pad = 1
        bos = 2
        test = 21959
        one = 112

        self.assertEqual(tokens.tolist(), [[bos, test, pad], [bos, test, one]])

    def test_tokens_to_strings(self):
        s = self._lm.tokens_to_strings(torch.Tensor([2, 21959, 112]).to(torch.int))
        self.assertEqual(s, ["</s>", "test", " 1"])

    def test_manage_state(self):
        # One lm.run was called at the setup. Call one more and make sure the object
        # returns values for the new state.
        self._lm.run(["one", "two", "three", "four"])
        self.assertEqual(self._lm.tokens().shape[0], 4)

    def test_residual_in_and_out(self):
        """
        Test that residual_in is a residual_out for the previous layer.
        """
        for layer in range(1, 12):
            prev_residual_out = self._lm.residual_out(layer - 1)
            residual_in = self._lm.residual_in(layer)
            diff = torch.max(torch.abs(residual_in - prev_residual_out)).item()
            self.assertLess(diff, self._eps, f"layer {layer}")

    def test_residual_plus_block(self):
        """
        Make sure that new residual = old residual + block output. Here, block is an ffn
        or attention. It's not that obvious because it could be that layer norm is
        applied after the block output, but before saving the result to residual.
        Luckily, this is not the case in TransformerLens, and we're relying on that.
        """
        layer = 3
        batch = 0
        pos = 0

        residual_in = self._lm.residual_in(layer)[batch][pos]
        residual_mid = self._lm.residual_after_attn(layer)[batch][pos]
        residual_out = self._lm.residual_out(layer)[batch][pos]
        ffn_out = self._lm.ffn_out(layer)[batch][pos]
        attn_out = self._lm.attention_output(batch, layer, pos)

        a = residual_mid
        b = residual_in + attn_out
        diff = torch.max(torch.abs(a - b)).item()
        self.assertLess(diff, self._eps, "attn")

        a = residual_out
        b = residual_mid + ffn_out
        diff = torch.max(torch.abs(a - b)).item()
        self.assertLess(diff, self._eps, "ffn")

    def test_tensor_shapes(self):
        # Not much we can do about the tensors, but at least check their shapes and
        # that they don't contain NaNs.
        vocab_size = 50272
        n_batch = 2
        n_tokens = 3
        d_model = 768
        d_hidden = d_model * 4
        n_heads = 12
        layer = 5

        device = self._lm.residual_in(0).device

        for name, tensor, expected_shape in [
            ("r_in", self._lm.residual_in(layer), [n_batch, n_tokens, d_model]),
            (
                "r_mid",
                self._lm.residual_after_attn(layer),
                [n_batch, n_tokens, d_model],
            ),
            ("r_out", self._lm.residual_out(layer), [n_batch, n_tokens, d_model]),
            ("logits", self._lm.logits(), [n_batch, n_tokens, vocab_size]),
            ("ffn_out", self._lm.ffn_out(layer), [n_batch, n_tokens, d_model]),
            (
                "decomposed_ffn_out",
                self._lm.decomposed_ffn_out(0, 0, 0),
                [d_hidden, d_model],
            ),
            ("neuron_activations", self._lm.neuron_activations(0, 0, 0), [d_hidden]),
            ("neuron_output", self._lm.neuron_output(0, 0), [d_model]),
            (
                "attention_matrix",
                self._lm.attention_matrix(0, 0, 0),
                [n_tokens, n_tokens],
            ),
            (
                "attention_output_per_head",
                self._lm.attention_output_per_head(0, 0, 0, 0),
                [d_model],
            ),
            (
                "attention_output",
                self._lm.attention_output(0, 0, 0),
                [d_model],
            ),
            (
                "decomposed_attn",
                self._lm.decomposed_attn(0, layer),
                [n_tokens, n_tokens, n_heads, d_model],
            ),
            (
                "unembed",
                self._lm.unembed(torch.zeros([d_model]).to(device), normalize=True),
                [vocab_size],
            ),
        ]:
            self.assertEqual(list(tensor.shape), expected_shape, name)
            self.assertFalse(torch.any(tensor.isnan()), name)


if __name__ == "__main__":
    unittest.main()
