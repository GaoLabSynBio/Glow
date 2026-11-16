"""Intuitive colours for fluorescence channels.

A channel name such as ``"EGFP-A"``, ``"PE (R-phycoerythrin)-A"``, ``"PE-Cy7-A"`` or
``"BV421-A"`` is mapped to one of six colour families that follow the dye's emission
colour, so an EGFP histogram is green, PE / mCherry are red, APC / Cy7 are dark red:

    purple   violet-excited dyes & blue FPs (DAPI, Hoechst, BV421, Pacific Blue, BFP, …)
    blue     cyan FPs and ~500 nm dyes (CFP, mTurquoise, AmCyan, BV510, V500, …)
    green    GFP family, FITC, Alexa 488, mNeonGreen, CFSE, …
    orange   yellow FPs and ~560 nm dyes (YFP, Venus, Citrine, mOrange, mKO, Alexa 555, …)
    red      PE, mCherry, DsRed, tdTomato, TRITC, Texas Red, PI, Cy3, BV605, …
    darkred  far-red: APC, PerCP, Cy5/Cy7 tandems, 7-AAD, DRAQ5, mKate2, iRFP, BV650+, …

Matching is by name first (ordered, most specific first), then by the emission
wavelength embedded in brand names (``BV605``, ``Alexa Fluor 647``, ``eFluor 450``) or in
a band-pass such as ``"525/50"``. Scatter and time channels map to ``None``.
"""
from __future__ import annotations

import colorsys
import re

FAMILIES = ("purple", "blue", "green", "orange", "red", "darkred")

FAMILY_COLORS = {
    "purple": "#8e44ad",
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "orange": "#f39c12",
    "red": "#d62728",
    "darkred": "#8b0000",
}


def _band(nm: float) -> str:
    """Colour family for an emission wavelength in nm."""
    if nm < 470:
        return "purple"
    if nm < 512:
        return "blue"
    if nm < 550:
        return "green"
    if nm < 585:
        return "orange"
    if nm < 630:
        return "red"
    return "darkred"


# Ordered (family, regex) pairs matched against the "spaced" form of the channel name:
# lower-case alphabetic and numeric tokens separated by single spaces, e.g.
#   "PE-Cy7-A"            -> "pe cy 7 a"
#   "PE (R-phycoerythrin)" -> "pe r phycoerythrin"
#   "mTagBFP2"            -> "mtagbfp 2"
# Specific tandems and far-red names come before the generic dye names they contain.
_NAME_RULES: list[tuple[str, str]] = [
    # --- far red / near-IR first (they contain shorter dye names) ---
    ("darkred", r"\bpe ?cy ?(5|5\.5|7)\b"),
    ("darkred", r"\bpe ?(alexa|af|vio|efluor|ef) ?(647|680|700|750|770|780)\b"),
    ("darkred", r"\bapc\b|\bapc[a-z]|allophycocyanin"),
    ("darkred", r"\bpercp"),
    ("darkred", r"\bcy ?(5|5\.5|7)\b"),
    ("darkred", r"\b7 ?aad\b|draq ?[57]\b|\bto ?pro ?3\b|sytox ?red|\bhelix ?np"),
    ("darkred", r"mkate|mplum|mneptune|mraspberry|katushka|turbofp ?6\d\d|crimson|mcardinal|mgarnet"),
    ("darkred", r"\birfp|\bmirfp|\bemirfp|\bifp\b|\w*rfp ?6\d\d\b"),
    ("darkred", r"zombie ?nir|near ?ir\b|\bnir\b|far ?red|\bir ?dye|\bbodipy ?6\d\d"),
    # --- red ---
    ("red", r"\bpe ?(cf|dazzle|texas|tx|tr) ?(594|red)?\b|\bpe ?(alexa|af|vio|efluor|ef) ?(610|615)\b"),
    ("red", r"\bpe\b|phycoerythrin"),
    ("red", r"mcherry|\bcherry\b|dsred|\w*rfp\b|tomato|mstrawberry|mscarlet|mruby|mapple|mrouge"),
    ("red", r"texas ?red|\btritc\b|rhodamine|propidium|\bpi\b|ethidium|\bethd|\bcy ?3(\.5)?\b"),
    ("red", r"zombie ?red|\bred\b"),
    # --- orange (yellow FPs, ~560 nm dyes) ---
    ("orange", r"\w*yfp\b|\bvenus\b|citrine|\bypet\b|\btopaz\b|zsyellow|mbanana|morange|\bmko ?\d?\b|kusabira"),
    ("orange", r"pacific ?orange|krome ?orange|\borange\b|\byellow\b"),
    # --- green ---
    ("green", r"\w*gfp\b|\bgfp|mneon|clover|emerald|azami|zsgreen|\bfitc\b|fluorescein|\bcfse\b|\bcfda"),
    ("green", r"calcein|sytox ?green|\bfam\b|oregon ?green|\bbodipy ?fl\b|zombie ?green|\bgreen\b"),
    # --- blue (cyan FPs) ---
    ("blue", r"\w*cfp\b|mturquoise|cerulean|amcyan|\bcyan\b|cypet|\bmtfp|\bteal\b|aquamarine|\baqua\b"),
    # --- purple (violet-excited, blue FPs, nuclear dyes) ---
    ("purple", r"\w*bfp\b|sirius|azurite|\bdapi\b|hoechst|pacific ?blue|cascade ?blue|marina ?blue"),
    ("purple", r"sytox ?blue|\bviolet\b|\buv\b|\bblue\b"),
]
_COMPILED = [(fam, re.compile(pat)) for fam, pat in _NAME_RULES]

# Brand prefixes whose number is the emission wavelength (BV605 emits at 605 nm).
_EMISSION_BRAND = re.compile(
    r"\b(?:bv|buv|byg|sb|bb|v|efluor|ef|qdot|ghost(?: dye)?(?: violet| red)?|super bright|"
    r"brilliant(?: violet| ultraviolet| blue)?|pacific|spark|fire|nova ?fluor|vio ?(?:blue|green)?|"
    r"star ?bright|realblue|realyellow|kiravia)\s?(\d{3})\b"
)
# Brand prefixes whose number is the excitation wavelength (Alexa Fluor 488 emits at ~519 nm).
_EXCITATION_BRAND = re.compile(r"\b(?:alexa(?: fluor)?|af|dylight|atto|cf)\s?(\d{3})\b")
_EXCITATION_TO_EMISSION_NM = 25
# Band-pass filter spec such as "525/50" or "FL1 530/30" (matched on the raw name, which keeps the slash).
_BANDPASS = re.compile(r"(?<!\d)(\d{3})\s*/\s*\d{2,3}(?!\d)")
# Detector-style names: a short letter code followed by a wavelength, e.g. "B525", "YG610", "R670".
_DETECTOR = re.compile(r"\b[a-z]{1,3} (\d{3})\b")

_SCATTER = re.compile(r"\b(fsc|ssc|time|event|tlsw|tmsw|width)\b|\bfs\b|\bss\b")


def _spaced(name: str) -> str:
    return " ".join(re.findall(r"[a-z]+|\d+(?:\.\d+)?", str(name).lower()))


def fluor_family(channel: str | None) -> str | None:
    """Return the colour family for a channel name, or None if it is not a fluorophore."""
    if not channel:
        return None
    s = _spaced(channel)
    if not s or _SCATTER.search(s):
        return None
    for fam, rx in _COMPILED:
        if rx.search(s):
            return fam
    for rx, text, offset in ((_EMISSION_BRAND, s, 0),
                             (_EXCITATION_BRAND, s, _EXCITATION_TO_EMISSION_NM),
                             (_BANDPASS, str(channel).lower(), 0),
                             (_DETECTOR, s, 0)):
        m = rx.search(text)
        if m:
            nm = int(m.group(1))
            if 380 <= nm <= 850:
                return _band(nm + offset)
    return None


def fluor_color(channel: str | None) -> str | None:
    """Base hex colour for a channel, or None."""
    fam = fluor_family(channel)
    return FAMILY_COLORS[fam] if fam else None


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    """Parse ``#rrggbb`` or Plotly-style ``rgb(r, g, b)`` into 0–1 floats."""
    h = str(h).strip()
    if h.lower().startswith("rgb"):
        nums = [float(n) for n in re.findall(r"[\d.]+", h)][:3]
        return tuple(min(1.0, n / 255.0) for n in nums)
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb)


def shades(base_hex: str, n: int) -> list[str]:
    """``n`` distinguishable shades of one hue, darkest first (n == 1 → the base colour)."""
    if n <= 1:
        return [base_hex]
    h, l, s = colorsys.rgb_to_hls(*_hex_to_rgb(base_hex))
    lo, hi = max(0.18, l - 0.14), min(0.78, l + 0.26)
    return [_rgb_to_hex(colorsys.hls_to_rgb(h, lo + (hi - lo) * i / (n - 1), s)) for i in range(n)]


def fluor_palette(channel: str | None, n: int) -> list[str] | None:
    """``n`` shades of the channel's colour, or None when the channel is not a fluorophore."""
    base = fluor_color(channel)
    return shades(base, n) if base else None


def muted_fill(hex_color: str, alpha: float = 0.4):
    """A half-transparent, slightly desaturated/lighter fill for an outline colour.

    Returns an ``rgba(r, g, b, a)`` string for Plotly; use ``muted_fill_rgba`` for matplotlib.
    """
    r, g, b, a = muted_fill_rgba(hex_color, alpha)
    return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})"


def muted_fill_rgba(hex_color: str, alpha: float = 0.4) -> tuple[float, float, float, float]:
    h, l, s = colorsys.rgb_to_hls(*_hex_to_rgb(hex_color))
    r, g, b = colorsys.hls_to_rgb(h, min(0.85, l + 0.12), s * 0.7)
    return (r, g, b, alpha)
