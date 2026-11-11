"""
Migrate ngso2gso parameter files from the old single-system format

    ngso2gso:
      ngso:
        n_co_channel: 8
        ...

to the multi-system format

    ngso2gso:
      ngso_systems:
        - label: ngso
          n_co_channel: 8
          ...

The rewrite is text-based so comments and YAML anchors are preserved.
Files are modified in place. Already-migrated files are skipped.

Usage:
    python scenarios/migrate_to_ngso_systems.py <file_or_dir> [...]

Directories are searched recursively for *.yaml / *.yml files.
"""
import re
import sys
from pathlib import Path

SECTION = "ngso2gso"
OLD_KEY = "ngso"
NEW_KEY = "ngso_systems"
DEFAULT_LABEL = "ngso"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_blank(line: str) -> bool:
    return line.strip() == ""


def migrate_text(text: str) -> tuple[str | None, str]:
    """
    Returns (new_text, status). new_text is None if nothing was changed;
    status explains why.
    """
    lines = text.splitlines(keepends=True)

    # Locate the top-level section
    try:
        sec_idx = next(
            i for i, ln in enumerate(lines)
            if re.match(rf"^{SECTION}:\s*(#.*)?$", ln.rstrip("\n"))
        )
    except StopIteration:
        return None, f"no top-level '{SECTION}:' section"

    # Section body: lines after it until the next top-level key
    sec_end = len(lines)
    child_indent = None
    for i in range(sec_idx + 1, len(lines)):
        ln = lines[i]
        if _is_blank(ln):
            continue
        ind = _indent(ln)
        if ind == 0 and not ln.lstrip().startswith("#"):
            sec_end = i
            break
        if child_indent is None and not ln.lstrip().startswith("#"):
            child_indent = ind

    if child_indent is None:
        return None, f"empty '{SECTION}' section"

    key_re = re.compile(r"^( *)(\w+):(.*)$")
    old_idx = None
    for i in range(sec_idx + 1, sec_end):
        m = key_re.match(lines[i].rstrip("\n"))
        if not m or len(m.group(1)) != child_indent:
            continue
        if m.group(2) == NEW_KEY:
            return None, "already migrated"
        if m.group(2) == OLD_KEY:
            rest = m.group(3).strip()
            if rest and not rest.startswith("#"):
                return None, (
                    f"'{OLD_KEY}:' uses inline/flow style, migrate it by hand"
                )
            old_idx = i

    if old_idx is None:
        return None, f"no '{SECTION}.{OLD_KEY}' key"

    # Block of the old key: following lines indented deeper than the key.
    # Comment lines count as part of the block only if indented deeper.
    blk_end = old_idx + 1
    last_content = old_idx
    for i in range(old_idx + 1, sec_end):
        ln = lines[i]
        if _is_blank(ln):
            continue
        if _indent(ln) <= child_indent:
            break
        last_content = i
    blk_end = last_content + 1

    body = lines[old_idx + 1:blk_end]
    body_indent = min(
        (_indent(ln) for ln in body
         if not _is_blank(ln) and not ln.lstrip().startswith("#")),
        default=None,
    )
    if body_indent is None:
        return None, f"'{SECTION}.{OLD_KEY}' has no content"

    newline = "\n" if lines[old_idx].endswith("\n") else ""
    key_line = (
        " " * child_indent + f"{NEW_KEY}:"
        + lines[old_idx].rstrip("\n")[child_indent + len(OLD_KEY) + 1:]
        + newline
    )
    new_block = [
        key_line,
        " " * body_indent + f"- label: {DEFAULT_LABEL}\n",
    ]
    for ln in body:
        new_block.append(ln if _is_blank(ln) else "  " + ln)

    new_lines = lines[:old_idx] + new_block + lines[blk_end:]
    return "".join(new_lines), "migrated"


def iter_files(paths: list[str]):
    for p in map(Path, paths):
        if p.is_dir():
            for ext in ("*.yaml", "*.yml"):
                yield from sorted(p.rglob(ext))
        elif p.is_file():
            yield p
        else:
            print(f"[WARN] {p} does not exist")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 1

    n_migrated = 0
    for f in iter_files(argv):
        new_text, status = migrate_text(f.read_text())
        if new_text is None:
            print(f"[SKIP] {f}: {status}")
            continue
        f.write_text(new_text)
        n_migrated += 1
        print(f"[MIGRATED] {f}")

    print(f"{n_migrated} file(s) migrated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
