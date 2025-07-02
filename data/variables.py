import re

PUNCT_TAGS = {"Ø": 0, ",": 1, ".": 2, "?": 3, "¿": 4}
PUNCT_END_TAGS = {"Ø": 0, ",": 1, ".": 2, "?": 3}
PUNCT_START_TAGS = {"Ø": 0, "¿": 1}
CAP_TAGS = {"lower": 0, "init": 1, "mix": 2, "upper": 3}
DEFAULT_PUNCT_MAP = {0: "", 1: ",", 2: ".", 3: "?", 4: "¿"}
ACCENT_RE = re.compile(r"[áéíóúÁÉÍÓÚñÑüÜ]")