"""Repair malformed DXF byte streams before handing them to ezdxf.

DXF is a line-pair format: a line holding a group code, then a line holding that
code's value. Nothing in the format escapes a newline inside a string value, so a
writer that emits one shifts every subsequent pair by one line and the whole file
becomes unparseable from that point on.

LibreDWG's ``dwg2dxf`` does exactly this with multi-line MTEXT/TEXT — an AutoCAD
note like::

    PROVIDE 150mm
     SERVICES CUT OUT AS PER MEP CONSULTANTS.

lands in the DXF as two physical lines, and ezdxf then reads
``" SERVICES CUT OUT AS PER MEP CONSULTANTS."`` where it expected an integer group
code. ezdxf's own ``recover`` module cannot fix it either: it repairs bad *tags*,
but this defect destroys the line *parity* that tag reading depends on.

The fix is to restore parity. We walk the file tracking whether the next line
should be a code or a value. When a line that should be a code is not an integer,
it must be the continuation of the value we just read, so we fold it back into
that value as a DXF ``\\P`` paragraph break — which is what the writer should have
emitted in the first place.

Working on bytes rather than text matters: real CAD files carry cp1252/latin-1
bytes and mojibake that would raise on strict UTF-8 decoding, and we must not
corrupt coordinates while cleaning up prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DxfRepairReport", "repair_dxf_bytes", "repair_dxf_file", "load_dxf"]

# Group codes are written right-aligned in a 3-char field, so leading/trailing
# whitespace is normal and must be stripped before the integer test.
_INT_RE = re.compile(rb"^[+-]?\d+$")

# Codes whose values are free text and therefore the only ones that can plausibly
# contain a stray newline. Restricting the fold to these avoids "repairing" a
# genuinely truncated file by gluing unrelated garbage onto a coordinate.
#   1   primary text value        2  name          3/4  extended text
#   7   text style name           300-309 arbitrary text
#   330+ are handles, not text, so they are deliberately excluded.
_TEXT_CODES = frozenset({1, 2, 3, 4, 7, *range(300, 310), *range(1000, 1010)})

_MAX_FOLD_LINES = 64  # A note longer than this signals real corruption, not a newline.


@dataclass
class DxfRepairReport:
    """What the repair pass changed, so ingest can surface provenance to the user."""

    folded_lines: int = 0
    """Stray continuation lines folded back into the preceding text value."""

    dropped_lines: int = 0
    """Unrecoverable lines discarded to keep the stream parseable."""

    samples: list[str] = field(default_factory=list)
    """First few folded fragments, for logging/debugging."""

    @property
    def clean(self) -> bool:
        return self.folded_lines == 0 and self.dropped_lines == 0

    def summary(self) -> str:
        if self.clean:
            return "DXF required no repair"
        return (
            f"DXF repaired: folded {self.folded_lines} stray text continuation line(s), "
            f"dropped {self.dropped_lines} unrecoverable line(s)"
        )


def _is_group_code(line: bytes) -> bool:
    return bool(_INT_RE.match(line.strip()))


def repair_dxf_bytes(raw: bytes) -> tuple[bytes, DxfRepairReport]:
    """Restore code/value line parity in a DXF byte stream.

    Returns the repaired bytes and a report of what changed. Line endings are
    normalised to CRLF, which is what the DXF spec calls for and what every
    consumer accepts.
    """
    report = DxfRepairReport()

    # Normalise endings first so the parity walk sees one line per record
    # regardless of whether the writer used CRLF, LF, or mixed them.
    lines = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")

    out: list[bytes] = []
    expecting_code = True
    current_code: int | None = None
    fold_run = 0

    for line in lines:
        if expecting_code:
            if _is_group_code(line):
                current_code = int(line.strip())
                out.append(line)
                expecting_code = False
                fold_run = 0
                continue

            # Not an integer where a code belongs: this is the tail of the value
            # we just emitted, split across a newline by the writer.
            if (
                out
                and current_code is not None
                and current_code in _TEXT_CODES
                and fold_run < _MAX_FOLD_LINES
            ):
                fragment = line.strip()
                if fragment:
                    # \P is the DXF paragraph break — the correct encoding for the
                    # newline the writer dropped.
                    out[-1] = out[-1] + rb"\P" + fragment
                    if len(report.samples) < 5:
                        report.samples.append(
                            fragment[:80].decode("utf-8", errors="replace")
                        )
                report.folded_lines += 1
                fold_run += 1
                continue

            # Nothing sensible to attach it to. Dropping keeps parity intact,
            # which is strictly better than aborting the whole file.
            if line.strip():
                report.dropped_lines += 1
            continue

        # Expecting a value: take the line verbatim. Values are allowed to look
        # like integers, so no validation is possible or wanted here.
        out.append(line)
        expecting_code = True

    return b"\r\n".join(out), report


def repair_dxf_file(src: Path | str, dst: Path | str) -> DxfRepairReport:
    """Repair ``src`` into ``dst``. Returns the repair report."""
    src, dst = Path(src), Path(dst)
    repaired, report = repair_dxf_bytes(src.read_bytes())
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(repaired)
    return report


def load_dxf(path: Path | str, *, work_dir: Path | str | None = None):
    """Load a DXF with escalating tolerance, returning ``(doc, notes)``.

    Tries, in order: strict ezdxf, ezdxf's own recover pass, then our parity
    repair followed by recover. Real CAD exports routinely need the last one.
    """
    import ezdxf
    import ezdxf.recover

    path = Path(path)
    notes: list[str] = []

    try:
        return ezdxf.readfile(str(path)), notes
    except Exception as exc:  # noqa: BLE001 - any parse failure escalates
        notes.append(f"strict parse failed ({type(exc).__name__}: {exc})")

    try:
        doc, auditor = ezdxf.recover.readfile(str(path))
        notes.append(
            f"recovered by ezdxf ({len(auditor.errors)} errors, {len(auditor.fixes)} fixes)"
        )
        return doc, notes
    except Exception as exc:  # noqa: BLE001
        notes.append(f"ezdxf recover failed ({type(exc).__name__}: {exc})")

    work_dir = Path(work_dir) if work_dir else path.parent
    repaired_path = work_dir / f"{path.stem}.repaired.dxf"
    report = repair_dxf_file(path, repaired_path)
    notes.append(report.summary())

    doc, auditor = ezdxf.recover.readfile(str(repaired_path))
    notes.append(
        f"recovered after repair ({len(auditor.errors)} errors, {len(auditor.fixes)} fixes)"
    )
    return doc, notes
