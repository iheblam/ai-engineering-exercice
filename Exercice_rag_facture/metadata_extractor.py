import re
from datetime import datetime
from pypdf import PdfReader


def read_value(text, pattern):
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return ""


def extract_metadata(file_path):
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    layout = "\n".join(
        page.extract_text(extraction_mode="layout") or "" for page in reader.pages
    )
    date_pattern = r"[A-Za-z]{3}\s+\d{1,2},?\s+\d{4}"
    metadata = {
        "code_facture": read_value(layout, r"^\s*#\s*(\S+)"),
        "date": "",
        "bill_to": read_value(text, r"Bill To\s*:\s*(.*?)Ship To\s*:"),
        "ship_to": read_value(text, rf"Ship To\s*:\s*(.*?)(?={date_pattern})"),
    }
    date = read_value(layout, rf"Date:\s*({date_pattern})")
    if date:
        metadata["date"] = datetime.strptime(
            date.replace(",", ""), "%b %d %Y"
        ).strftime("%Y-%m-%d")

    for field, label in [
        ("total", "Total"),
        ("shipping", "Shipping"),
        ("discount", r"Discount(?:\s*\([^)]*\))?"),
        ("amount", "Subtotal"),
    ]:
        value = read_value(
            layout,
            rf"^\s*{label}\s*:\s*\$?([\d,]+\.\d{{2}})",
        )
        metadata[field] = float(value.replace(",", "")) if value else None

    return metadata
