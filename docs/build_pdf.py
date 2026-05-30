"""Build a simple PDF from the project Markdown documentation.

This avoids external dependencies such as pandoc or LaTeX. It intentionally
supports only the plain Markdown subset used by psi_sapu_documentation.md.
"""

from __future__ import annotations

from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "psi_sapu_documentation.md"
OUTPUT = ROOT / "psi_sapu_documentation.pdf"

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT = 54
TOP = 738
BOTTOM = 54
LINE_HEIGHT = 12
BODY_SIZE = 9
HEADING_SIZE = 14
TITLE_SIZE = 18
MAX_CHARS = 92


def pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .encode("latin-1", "replace")
        .decode("latin-1")
    )


def render_markdown_lines(text: str) -> list[tuple[str, int, bool]]:
    rendered: list[tuple[str, int, bool]] = []
    in_code = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            rendered.append(("", BODY_SIZE, False))
            continue
        if in_code:
            wrapped = textwrap.wrap(line, width=MAX_CHARS) or [""]
            rendered.extend((part, BODY_SIZE, False) for part in wrapped)
            continue
        if line.startswith("# "):
            rendered.append((line[2:].strip(), TITLE_SIZE, True))
            rendered.append(("", BODY_SIZE, False))
            continue
        if line.startswith("## "):
            rendered.append(("", BODY_SIZE, False))
            rendered.append((line[3:].strip(), HEADING_SIZE, True))
            continue
        if line.startswith("### "):
            rendered.append(("", BODY_SIZE, False))
            rendered.append((line[4:].strip(), BODY_SIZE + 2, True))
            continue
        if not line:
            rendered.append(("", BODY_SIZE, False))
            continue

        indent = ""
        content = line
        if line.startswith("- "):
            indent = "  "
            content = "* " + line[2:]
        elif line[0:2].isdigit() and ". " in line[:4]:
            indent = "  "
            content = line

        wrapped = textwrap.wrap(
            content,
            width=MAX_CHARS - len(indent),
            subsequent_indent="  ",
        ) or [""]
        rendered.extend((indent + part, BODY_SIZE, False) for part in wrapped)
    return rendered


def paginate(lines: list[tuple[str, int, bool]]) -> list[list[tuple[str, int, bool]]]:
    pages: list[list[tuple[str, int, bool]]] = []
    page: list[tuple[str, int, bool]] = []
    y = TOP
    for line in lines:
        _, size, _ = line
        height = max(LINE_HEIGHT, size + 3)
        if y - height < BOTTOM and page:
            pages.append(page)
            page = []
            y = TOP
        page.append(line)
        y -= height
    if page:
        pages.append(page)
    return pages


def page_stream(page: list[tuple[str, int, bool]], page_number: int) -> bytes:
    commands = ["BT"]
    y = TOP
    for text, size, bold in page:
        font = "F2" if bold else "F1"
        escaped = pdf_escape(text)
        commands.append(f"/{font} {size} Tf")
        commands.append(f"1 0 0 1 {LEFT} {y} Tm")
        commands.append(f"({escaped}) Tj")
        y -= max(LINE_HEIGHT, size + 3)
    commands.append("/F1 8 Tf")
    commands.append(f"1 0 0 1 {PAGE_WIDTH - 90} 30 Tm")
    commands.append(f"(Page {page_number}) Tj")
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("latin-1", "replace")


def build_pdf(pages: list[list[tuple[str, int, bool]]]) -> bytes:
    pages_obj = 2
    font_regular = 3
    font_bold = 4
    objects: list[bytes | None] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]

    page_object_ids: list[int] = []
    for page_number, page in enumerate(pages, start=1):
        stream = page_stream(page, page_number)
        stream_obj = len(objects) + 1
        objects.append(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
            + stream
            + b"endstream"
        )
        page_obj = len(objects) + 1
        objects.append(
            (
                f"<< /Type /Page /Parent {pages_obj} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> >> "
                f"/Contents {stream_obj} 0 R >>"
            ).encode("ascii")
        )
        page_object_ids.append(page_obj)

    kids = " ".join(f"{obj_id} 0 R" for obj_id in page_object_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode(
        "ascii"
    )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        if obj is None:
            raise RuntimeError(f"PDF object {index} was not populated.")
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    pages = paginate(render_markdown_lines(source))
    OUTPUT.write_bytes(build_pdf(pages))
    print(f"Wrote {OUTPUT} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
