import pytest

from glow.plotting.fluor_colors import (
    FAMILY_COLORS, fluor_family, fluor_palette, muted_fill, muted_fill_rgba, shades,
)


@pytest.mark.parametrize("name, family", [
    # green
    ("EGFP-A", "green"), ("GFP", "green"), ("sfGFP-A", "green"), ("TurboGFP", "green"),
    ("FITC-A", "green"), ("Alexa Fluor 488-A", "green"), ("mNeonGreen", "green"),
    ("CFSE", "green"), ("BB515-A", "green"), ("GFP (B525)", "green"), ("FL1 530/30", "green"),
    # orange
    ("EYFP-A", "orange"), ("Venus", "orange"), ("mCitrine", "orange"), ("YPet", "orange"),
    ("mOrange", "orange"), ("mKO2", "orange"), ("Pacific Orange", "orange"), ("Alexa 555", "orange"),
    ("BV570", "orange"),
    # red
    ("PE (R-phycoerythrin)-A", "red"), ("PE-A", "red"), ("mCherry-A", "red"), ("DsRed", "red"),
    ("tdTomato", "red"), ("mScarlet-I", "red"), ("TagRFP", "red"), ("mRFP1", "red"),
    ("PE-Texas Red-A", "red"), ("PE-CF594", "red"), ("PE-Dazzle 594", "red"), ("PI-A", "red"),
    ("Propidium Iodide", "red"), ("TRITC", "red"), ("Cy3", "red"), ("BV605-A", "red"),
    ("Alexa Fluor 594", "red"), ("LIVE/DEAD Fixable Red", "red"), ("Zombie Red", "red"),
    # dark red
    ("APC-A", "darkred"), ("APC-Cy7-A", "darkred"), ("APC-H7", "darkred"), ("APC-Fire 750", "darkred"),
    ("PE-Cy7-A", "darkred"), ("PE-Cy5", "darkred"), ("PE-Cy5.5", "darkred"), ("PerCP-Cy5.5-A", "darkred"),
    ("PerCP", "darkred"), ("7-AAD", "darkred"), ("DRAQ5", "darkred"), ("DRAQ7", "darkred"),
    ("TO-PRO-3", "darkred"), ("Cy5", "darkred"), ("Cy7", "darkred"), ("mKate2", "darkred"),
    ("iRFP670", "darkred"), ("miRFP703", "darkred"), ("TagRFP657", "darkred"), ("E2-Crimson", "darkred"),
    ("BV650-A", "darkred"), ("BV711", "darkred"), ("BV786", "darkred"), ("Alexa Fluor 647-A", "darkred"),
    ("Alexa Fluor 700", "darkred"), ("eFluor 780", "darkred"), ("Zombie NIR", "darkred"),
    ("LIVE/DEAD Fixable Near-IR", "darkred"), ("LIVE/DEAD Far Red", "darkred"), ("PE-Alexa 700", "darkred"),
    ("R670/30", "darkred"),
    # blue
    ("ECFP", "blue"), ("mTurquoise2", "blue"), ("Cerulean", "blue"), ("AmCyan-A", "blue"),
    ("BV510-A", "blue"), ("V500", "blue"), ("Zombie Aqua", "blue"), ("eFluor 506", "blue"),
    ("BUV496", "blue"),
    # purple
    ("DAPI-A", "purple"), ("Hoechst 33342", "purple"), ("BV421-A", "purple"), ("Pacific Blue-A", "purple"),
    ("EBFP2", "purple"), ("mTagBFP2", "purple"), ("TagBFP", "purple"), ("eFluor 450", "purple"),
    ("V450-A", "purple"), ("Alexa Fluor 405", "purple"), ("Super Bright 436", "purple"), ("SB436", "purple"),
    ("BUV395", "purple"), ("Zombie Violet", "purple"), ("LIVE/DEAD Fixable Violet", "purple"),
    ("Ghost Dye Violet 450", "purple"), ("Qdot 450", "purple"),
])
def test_family_mapping(name, family):
    assert fluor_family(name) == family, name


@pytest.mark.parametrize("name", ["FSC-A", "SSC-H", "FSC 488/10-A", "SSC 488/10-W", "TIME", "Time",
                                  "Event Info", "TLSW", "", None, "Unknown-A", "FL1-A"])
def test_non_fluorophores(name):
    assert fluor_family(name) is None


def test_palette_and_shades():
    assert fluor_palette("FSC-A", 3) is None
    assert fluor_palette("EGFP-A", 1) == [FAMILY_COLORS["green"]]
    pal = fluor_palette("EGFP-A", 4)
    assert len(pal) == 4 and len(set(pal)) == 4
    assert all(p.startswith("#") and len(p) == 7 for p in pal)
    assert shades("#2ca02c", 1) == ["#2ca02c"]


def test_muted_fill():
    s = muted_fill("#2ca02c", alpha=0.4)
    assert s.startswith("rgba(") and s.endswith(", 0.4)")
    r, g, b, a = muted_fill_rgba("#d62728", 0.35)
    assert a == 0.35 and 0 <= r <= 1 and g > 0.15  # lighter / less saturated than pure red
