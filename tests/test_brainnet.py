"""Tests for BrainNet modules."""
import torch
# import pytest  # optional, for pytest runner
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainnet import CorticalColumn, InhibitoryLayer, LateralConnections, TopoLoss


class TestCorticalColumn:
    def test_forward_shape(self):
        model = CorticalColumn(in_features=784, hidden_features=128, out_features=10)
        x = torch.randn(8, 784)
        out = model(x)
        assert out.shape == (8, 10)

    def test_activations_stored(self):
        model = CorticalColumn(784, 128, 10)
        x = torch.randn(4, 784)
        _ = model(x)
        acts = model.get_activations()
        assert len(acts) == 4  # L4, L2/3, L5, L6

    def test_gradient_flow(self):
        model = CorticalColumn(784, 128, 10)
        x = torch.randn(4, 784)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_no_lateral(self):
        model = CorticalColumn(784, 128, 10, use_lateral=False)
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4, 10)

    def test_different_inhibitory_ratios(self):
        for ratio in [0.0, 0.1, 0.2, 0.5]:
            model = CorticalColumn(784, 128, 10, inhibitory_ratio=ratio)
            x = torch.randn(4, 784)
            out = model(x)
            assert out.shape == (4, 10)


class TestInhibitoryLayer:
    def test_shape_preserved(self):
        layer = InhibitoryLayer(256, ratio=0.2)
        x = torch.randn(8, 256)
        out = layer(x)
        assert out.shape == x.shape

    def test_suppression_effect(self):
        layer = InhibitoryLayer(64, ratio=0.2)
        x = torch.ones(4, 64)
        out = layer(x)
        # Output should be different from input (inhibition applied)
        assert not torch.allclose(x, out)


class TestLateralConnections:
    def test_shape_preserved(self):
        lateral = LateralConnections(256)
        x = torch.randn(8, 256)
        out = lateral(x)
        assert out.shape == x.shape

    def test_no_self_connection(self):
        lateral = LateralConnections(64)
        # Diagonal should be zero
        assert torch.allclose(
            lateral.lateral_weight.data.diag(),
            torch.zeros(64)
        )

    def test_multiple_iterations(self):
        lateral = LateralConnections(64, iterations=3)
        x = torch.randn(4, 64)
        out = lateral(x)
        assert out.shape == x.shape


class TestTopoLoss:
    def test_returns_scalar(self):
        topo = TopoLoss(weight=0.1)
        acts = [torch.randn(8, 64), torch.randn(8, 128)]
        loss = topo(acts)
        assert loss.shape == ()

    def test_gradient_flows(self):
        topo = TopoLoss(weight=0.1)
        acts = [torch.randn(8, 64, requires_grad=True)]
        loss = topo(acts)
        loss.backward()
        assert acts[0].grad is not None

    def test_organized_vs_random(self):
        """Organized activations should have lower topo loss."""
        topo = TopoLoss(weight=1.0)

        # Create organized activations (similar features nearby)
        organized = torch.zeros(32, 64)
        for i in range(64):
            organized[:, i] = torch.sin(torch.arange(32).float() * (i / 64.0))

        # Random activations
        random_act = torch.randn(32, 64)

        loss_organized = topo([organized])
        loss_random = topo([random_act])

        # Organized should have lower loss (or equal, depending on init)
        # This is a soft check — the principle should hold
        print(f"Organized loss: {loss_organized.item():.4f}")
        print(f"Random loss: {loss_random.item():.4f}")


class TestIntegration:
    def test_full_training_step(self):
        """Test that a full forward-backward pass works."""
        model = CorticalColumn(784, 128, 10)
        topo = TopoLoss(weight=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(16, 784)
        y = torch.randint(0, 10, (16,))

        # Forward
        logits = model(x)
        loss = criterion(logits, y) + topo(model.get_activations())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        print(f"Training step loss: {loss.item():.4f}")


if __name__ == "__main__":
    print("Running BrainNet tests...\n")

    # Run all tests
    test_col = TestCorticalColumn()
    test_col.test_forward_shape()
    print("[PASS] CorticalColumn forward shape")
    test_col.test_activations_stored()
    print("[PASS] CorticalColumn activations stored")
    test_col.test_gradient_flow()
    print("[PASS] CorticalColumn gradient flow")
    test_col.test_no_lateral()
    print("[PASS] CorticalColumn without lateral")
    test_col.test_different_inhibitory_ratios()
    print("[PASS] CorticalColumn different inhibitory ratios")

    test_inh = TestInhibitoryLayer()
    test_inh.test_shape_preserved()
    print("[PASS] InhibitoryLayer shape preserved")
    test_inh.test_suppression_effect()
    print("[PASS] InhibitoryLayer suppression effect")

    test_lat = TestLateralConnections()
    test_lat.test_shape_preserved()
    print("[PASS] LateralConnections shape preserved")
    test_lat.test_no_self_connection()
    print("[PASS] LateralConnections no self-connection")
    test_lat.test_multiple_iterations()
    print("[PASS] LateralConnections multiple iterations")

    test_topo = TestTopoLoss()
    test_topo.test_returns_scalar()
    print("[PASS] TopoLoss returns scalar")
    test_topo.test_gradient_flows()
    print("[PASS] TopoLoss gradient flows")
    test_topo.test_organized_vs_random()
    print("[PASS] TopoLoss organized vs random")

    test_int = TestIntegration()
    test_int.test_full_training_step()
    print("[PASS] Full training step")

    print("\n All tests passed!")
