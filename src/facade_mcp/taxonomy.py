"""Canonical business classes, independent of legacy checkpoint class IDs."""

CLASS_NAMES = {
    "C": "Crack", "P": "Peeling", "S": "Spalling", "T": "RustStain",
    "R": "Corrosion", "D": "Contamination", "V": "Vegetation",
    "L": "Leakage", "E": "Efflorescence",
}


def label_key(value):
    return "".join(c for c in str(value).lower() if c.isalnum())


EFFLORESCENCE_LABELS = {"efflorescence", "白华", "白華", "泛碱", "泛鹼", "盐析"}


def canonical_code(code, *labels):
    if code == "M":
        return "C"
    if code in (None, "L") and any(label_key(label) in EFFLORESCENCE_LABELS for label in labels):
        return "E"
    return code


def normalize_class(record):
    """Keep source labels when canonicalizing a new record; never mutate input."""
    result = dict(record)
    old = record.get("class_code")
    code = canonical_code(old, record.get("class_name", ""),
                          record.get("original_class_name", ""), record.get("class_subtype", ""))
    if isinstance(code, str) and code in CLASS_NAMES:
        if old != code:
            result.setdefault("original_class_code", old)
        if record.get("class_name") and record["class_name"] != CLASS_NAMES[code]:
            result.setdefault("original_class_name", record["class_name"])
        result.update(class_code=code, class_name=CLASS_NAMES[code])
        if code == "E" and label_key(result.get("class_subtype", "")) in EFFLORESCENCE_LABELS:
            result.pop("class_subtype", None)
    return result


def opencv_code(record):
    # Upstream L is a generic stain/deposit baseline, not an E-specific model.
    return "L" if record["class_code"] == "E" else record["class_code"]
