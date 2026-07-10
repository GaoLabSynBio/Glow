import pytest

pytest.importorskip("cytoflow")
from glow.gating.manager import GatingManager


class FakeExp:
    def __init__(self, channels):
        self.channels = channels


def test_exact_and_case_insensitive():
    gm = GatingManager()
    exp = FakeExp(["FSC-A", "SSC-A", "FSC-H"])
    assert gm._resolve_channel(exp, "FSC-A") == "FSC-A"
    assert gm._resolve_channel(exp, "fsc-a") == "FSC-A"


def test_variant_fallback_area_naming():
    gm = GatingManager()
    exp = FakeExp(["FSC-Area", "SSC-Area"])
    # "FSC-A" should resolve to the "FSC-Area" style name on this instrument.
    assert gm._resolve_channel(exp, "FSC-A") == "FSC-Area"
    assert gm._resolve_channel(exp, "SSC-A") == "SSC-Area"
