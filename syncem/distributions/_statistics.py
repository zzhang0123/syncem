"""Identity of placed distributions for grouping specified populations."""

from ..base._fingerprint import NotCacheable, digest
from ..base._function_fingerprint import function_fingerprint
from ._common import Law


def factor_record(factor):
    """Compare placed declarations without built-in display-unit metadata."""
    if hasattr(factor, "record"):
        record = factor.record()
        if "fingerprint" in record and record["fingerprint"] is None:
            return ("declaration", id(factor))
        kind = type(factor)
        if (
            kind.__module__.startswith("syncem.distributions.")
            and kind.record is Law.record
        ):
            record = {
                name: value
                for name, value in record.items()
                if name not in ("given_unit", "fingerprint")
            }
            record["parameters"] = {
                name: value
                for name, value in record["parameters"].items()
                if name != "given_unit"
            }
            return {**record, "fingerprint": digest(record)}
        return record
    if callable(factor):
        try:
            return ("function", function_fingerprint(factor))
        except NotCacheable:
            return ("declaration", id(factor))
    return factor


def factor_key(factor):
    """Hash a placed declaration or keep its identity when it cannot be fingerprinted."""
    try:
        return (type(factor), digest(factor_record(factor)))
    except (NotCacheable, TypeError, ValueError):
        return ("identity", id(factor))
