import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dv_training_utils import DisplacedVertexGNN  # noqa: E402


class GATv2EdgeAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.x = torch.randn(9, 7)
        self.edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 4, 6],
                [1, 2, 0, 4, 5, 3, 7, 8, 6, 2, 3, 8],
            ],
            dtype=torch.long,
        )
        self.edge_attr = torch.randn(self.edge_index.shape[1], 5)
        self.batch = torch.tensor(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            dtype=torch.long,
        )
        self.node_is_muon = torch.tensor(
            [True, True, False, True, False, False, True, False, False],
        )

    def build_model(self, **kwargs):
        return DisplacedVertexGNN(
            xdim=7,
            edim=5,
            hdim=16,
            n_layers=2,
            dropout=0.0,
            layer_type="gat_residual",
            gat_heads=4,
            pool="meanmax",
            **kwargs,
        )

    def forward(self, model, edge_attr=None):
        return model(
            self.x,
            self.edge_index,
            self.edge_attr if edge_attr is None else edge_attr,
            batch=self.batch,
            node_is_muon=self.node_is_muon,
        )

    def test_gatv2_uses_attention_and_message_parameters(self):
        model = self.build_model(gatv2_edge_attn=True)
        output = self.forward(model)

        self.assertEqual(tuple(output.shape), (3,))
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()

        expected_parameter_fragments = (
            "gatv2_attn_src.weight",
            "gatv2_attn_dst.weight",
            "gatv2_attn_edge.weight",
            "gatv2_attn_score",
            "gat.linear.weight",
            "gatv2_msg_dst.weight",
            "gatv2_msg_edge.weight",
        )
        grads = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if any(fragment in name for fragment in expected_parameter_fragments)
        }
        self.assertTrue(grads)
        self.assertTrue(all(grad is not None for grad in grads.values()))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads.values()))
        unused = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(unused, [], msg=f"DDP-unsafe unused parameters: {unused}")

    def test_gatv2_output_changes_when_one_edge_changes(self):
        model = self.build_model(gatv2_edge_attn=True).eval()
        changed_edge_attr = self.edge_attr.clone()
        changed_edge_attr[0] += torch.tensor([2.0, -1.0, 0.5, 1.5, -0.75])

        original = self.forward(model)
        changed = self.forward(model, changed_edge_attr)

        self.assertFalse(torch.allclose(original, changed, rtol=0.0, atol=1e-8))

    def test_attention_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.build_model(gat_edge_attn=True, gatv2_edge_attn=True)

    def test_legacy_modes_have_no_new_checkpoint_parameters(self):
        for kwargs in ({}, {"gat_edge_attn": True}):
            model = self.build_model(**kwargs)
            self.assertFalse(
                any("gatv2_" in key for key in model.state_dict()),
                msg=f"Unexpected GATv2 keys for legacy mode {kwargs}",
            )
            clone = self.build_model(**kwargs)
            clone.load_state_dict(model.state_dict(), strict=True)

    def test_gatv2_checkpoint_round_trip_is_strict(self):
        model = self.build_model(gatv2_edge_attn=True)
        clone = self.build_model(gatv2_edge_attn=True)
        clone.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(self.forward(model), self.forward(clone)))


if __name__ == "__main__":
    unittest.main()
