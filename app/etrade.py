"""In-memory import of text-based E*TRADE statement PDFs."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_EXTRACTED_TEXT_CHARS = 2_000_000
MAX_STRUCTURE_LINES = 120


class StatementPdfError(ValueError):
    """A PDF could not be read as a text-based brokerage statement."""


TRANSFER_STATUSES = ("Scheduled", "Complete", "In process", "Cancelled", "Rejected", "Failed", "Action required")
STRUCTURE_LABELS = {
    "account", "activity", "allocation", "amount", "annual", "asset", "available", "balance", "beginning",
    "cash", "change", "closing", "cost", "date", "description", "dividend", "estimated", "ending", "equity",
    "fixed", "fund", "gain", "gross", "income", "interest", "investment", "last", "loss", "market", "maturity",
    "net", "open", "option", "other", "payment", "performance", "portfolio", "position", "price", "purchase",
    "quantity", "rate", "reinvested", "security", "sell", "share", "shares", "symbol", "tax", "ticker", "total",
    "transaction", "type", "value", "withdrawal", "withdrawals", "year", "yield", "ytd", "summary", "holdings",
    "transactions", "details", "summary", "statement", "page", "transfer", "contribution", "distribution", "fee",
    "fees", "debit", "credit", "deposit", "withdraw", "short", "long", "term", "mutual", "stock", "bond",
    "description", "realized", "unrealized", "percent", "percentage", "change", "current", "prior",
}


def analyze_statement_text(text: str) -> dict:
    """Extract only the safe, structured fields available in a transfer-activity export."""
    if "transfer activity" not in text.lower():
        return {"document_type": "statement_text", "transactions": [], "summary": None}

    status_map = {
        icon: next(status for status in TRANSFER_STATUSES if status.lower() == label.lower())
        for icon, label in re.findall(
            r"([\ue000-\uf8ff]+)\s*(Scheduled|Complete|In process|Cancelled|Rejected|Failed|Action required)",
            text,
            flags=re.IGNORECASE,
        )
    }
    transactions = []
    for line in text.splitlines():
        date_match = re.search(r"(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+", line)
        amount_match = re.search(r"\$(?P<amount>\d[\d,]*(?:\.\d{2})?)", line)
        if not date_match or not amount_match or amount_match.start() < date_match.end():
            continue

        try:
            transaction_date = datetime.strptime(date_match.group("date"), "%m/%d/%Y").date().isoformat()
        except ValueError:
            continue

        marker = line[:date_match.start()].strip().split()
        icon = marker[-1] if marker else ""
        transactions.append({
            "date": transaction_date,
            "type": "Transfer",
            "amount": float(amount_match.group("amount").replace(",", "")),
            "status": status_map.get(icon, "Unknown"),
        })

    summary = None
    if transactions:
        summary = {
            "transaction_count": len(transactions),
            "total_amount": round(sum(item["amount"] for item in transactions), 2),
            "completed_amount": round(sum(item["amount"] for item in transactions if item["status"] == "Complete"), 2),
            "scheduled_amount": round(sum(item["amount"] for item in transactions if item["status"] == "Scheduled"), 2),
        }
    return {"document_type": "transfer_activity", "transactions": transactions, "summary": summary}


def _unique_labeled_amount(text: str, labels: tuple[str, ...]) -> float | None:
    alternatives = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    label_pattern = re.compile(rf"\b(?:{alternatives})\b", flags=re.IGNORECASE)
    amount_pattern = re.compile(r"\(?-?\s*\$?\s*\d[\d,]*(?:\.\d{2})?\)?")
    values = set()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        for match in label_pattern.finditer(line):
            remainder = line[match.end():].strip(" \t:-")
            amounts = amount_pattern.findall(remainder)
            if not amounts and not remainder and index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                next_amounts = amount_pattern.findall(next_line)
                if len(next_amounts) == 1 and not amount_pattern.sub("", next_line).strip(" \t:$(),-"):
                    amounts = next_amounts
            if len(amounts) != 1:
                continue
            amount_text = amounts[0].replace("$", "").replace(",", "").replace(" ", "")
            is_negative = amount_text.startswith("(") and amount_text.endswith(")")
            amount_text = amount_text.strip("()")
            amount = float(amount_text)
            values.add(round(-amount if is_negative else amount, 2))
    return next(iter(values)) if len(values) == 1 else None


def _fidelity_overview_diagnostics(text: str) -> dict:
    first_page = re.split(r"--- Page 2 ---", text, maxsplit=1, flags=re.IGNORECASE)[0]
    label_patterns = (
        re.compile(r"Your\s+Portfolio\s+Value\s*:?", flags=re.IGNORECASE),
        re.compile(r"Portfolio\s+Change\s+from\s+Last\s+Period\s*:?", flags=re.IGNORECASE),
        re.compile(r"Total\s+Including\s+Other\s+Holdings(?:\s*\d+)?\s*:?", flags=re.IGNORECASE),
    )
    labels = [pattern.search(first_page) for pattern in label_patterns]
    labels_found = sum(match is not None for match in labels)
    if labels_found == len(label_patterns):
        labels.sort(key=lambda match: match.start())
        values_start = labels[-1].end()
        boundary = re.search(
            r"\bBeginning\s+Portfolio\s+Value\b",
            first_page[values_start:],
            flags=re.IGNORECASE,
        )
    else:
        values_start = 0
        boundary = None

    amount_count = 0
    if boundary:
        values_end = values_start + boundary.start()
        amount_count = len(re.findall(
            r"(?<![\w.])\(?-?\s*\$?\s*-?\d[\d,]*\.\d{2}\)?",
            first_page[values_start:values_end],
        ))
    return {
        "labels_found": labels_found,
        "boundary_found": boundary is not None,
        "amount_count": amount_count,
    }


def _fidelity_portfolio_sections(text: str) -> dict:
    summary = {
        "income_this_period": None,
        "income_year_to_date": None,
        "top_holdings_value": None,
        "top_holdings_percent": None,
    }
    summary_heading = re.search(r"\bPortfolio\s+Summary\b", text, flags=re.IGNORECASE)
    if not summary_heading:
        return summary

    income_match = re.search(r"\bIncome\s+Summary\b", text[summary_heading.end():], flags=re.IGNORECASE)
    if not income_match:
        return summary
    income_start = summary_heading.end() + income_match.end()
    top_holdings_match = re.search(r"\bTop\s+Holdings\b", text[income_start:], flags=re.IGNORECASE)
    if not top_holdings_match:
        return summary
    top_holdings_start = income_start + top_holdings_match.start()
    income_text = text[income_start:top_holdings_start]
    income_totals = list(re.finditer(r"\bTotal\b", income_text, flags=re.IGNORECASE))
    if income_totals:
        totals_text = income_text[income_totals[-1].end():]
        income_amounts = re.findall(
            r"(?<![\w.])\$?\s*\d[\d,]*\.\d{2}",
            totals_text,
        )
        if len(income_amounts) == 2:
            summary["income_this_period"], summary["income_year_to_date"] = (
                round(float(amount.replace("$", "").replace(",", "").strip()), 2)
                for amount in income_amounts
            )

    top_text = text[top_holdings_start:]
    top_total = re.search(r"\bTotal\b", top_text, flags=re.IGNORECASE)
    if top_total:
        total_text = top_text[top_total.end():]
        amount_match = re.search(r"(?<![\w.])\$?\s*\d[\d,]*(?:\.\d{2})?", total_text)
        percent_match = re.search(r"\b(\d{1,3})\s*%", total_text)
        if amount_match:
            summary["top_holdings_value"] = float(
                amount_match.group().replace("$", "").replace(",", "").strip()
            )
        if percent_match:
            summary["top_holdings_percent"] = float(percent_match.group(1))

    return summary


def _parse_first_amount(value: str) -> float | None:
    match = re.search(r"(?<![\w.])(-?\s*\$?\s*-?\d[\d,]*(?:\.\d{1,2})?)", value)
    if not match:
        return None
    amount_text = match.group(1).replace("$", "").replace(",", "").replace(" ", "")
    return round(float(amount_text), 2)


def _fidelity_account_summaries(text: str) -> list[dict]:
    summaries = []
    pages = re.split(r"--- Page \d+ ---", text, flags=re.IGNORECASE)
    account_heading = re.compile(r"\bAccount\s+Summary\b", flags=re.IGNORECASE)
    field_labels = {
        "account_value": re.compile(r"^\s*Account\s+Value\s*:?\s*(.*)$", re.IGNORECASE),
        "change_in_account_value": re.compile(r"^\s*Change\s+in\s+Account\s+Value\s*:?\s*(.*)$", re.IGNORECASE),
        "beginning_value": re.compile(r"^\s*Beginning\s+Account\s+Value\s*:?\s*(.*)$", re.IGNORECASE),
        "additions": re.compile(r"^\s*Additions\s*:?\s*(.*)$", re.IGNORECASE),
        "subtractions": re.compile(r"^\s*Subtractions\s*:?\s*(.*)$", re.IGNORECASE),
        "transaction_costs": re.compile(r"^\s*Transaction\s+Costs,\s*Fees\s*&\s*Charges\s*:?\s*(.*)$", re.IGNORECASE),
        "investment_change": re.compile(r"^\s*Change\s+in\s+Investment\s+Value\s*\*?\s*:?\s*(.*)$", re.IGNORECASE),
        "ending_value": re.compile(r"^\s*Ending\s+Account\s+Value(?!\s+Incl\.?\s+AI)\s*\*?\s*:?\s*(.*)$", re.IGNORECASE),
        "accrued_interest": re.compile(r"^\s*Accrued\s+Interest\s*\(AI\)\s*:?\s*(.*)$", re.IGNORECASE),
    }

    for page in pages:
        heading = account_heading.search(page)
        if not heading:
            continue
        account_text = page[heading.end():]
        lines = account_text.splitlines()
        summary = {field: None for field in field_labels}
        for index, line in enumerate(lines):
            for field, pattern in field_labels.items():
                match = pattern.match(line)
                if not match:
                    continue
                amount = _parse_first_amount(match.group(1))
                if amount is None and index + 1 < len(lines):
                    amount = _parse_first_amount(lines[index + 1])
                if amount is not None:
                    summary[field] = amount

        top_match = re.search(r"\bTop\s+Holdings\b", account_text, flags=re.IGNORECASE)
        income_match = re.search(r"\bIncome\s+Summary\b", account_text, flags=re.IGNORECASE)
        if top_match:
            top_end = income_match.start() if income_match and income_match.start() > top_match.end() else len(account_text)
            top_text = account_text[top_match.end():top_end]
            totals = list(re.finditer(r"(?im)^\s*Total\b(.*)$", top_text))
            if totals:
                total_line = totals[-1].group(1)
                summary["top_holdings_value"] = _parse_first_amount(total_line)
                percent = re.search(r"\b(\d{1,3})\s*%", total_line)
                if percent:
                    summary["top_holdings_percent"] = float(percent.group(1))
                else:
                    summary["top_holdings_percent"] = None
            else:
                summary["top_holdings_value"] = None
                summary["top_holdings_percent"] = None
        else:
            summary["top_holdings_value"] = None
            summary["top_holdings_percent"] = None

        if income_match:
            income_text = account_text[income_match.end():]
            totals = list(re.finditer(r"(?im)^\s*Total\b(.*)$", income_text))
            if totals:
                values = re.findall(r"(?<![\w.])\$?\s*-?\d[\d,]*\.\d{2}", totals[-1].group(1))
                summary["income_this_period"] = (
                    round(float(values[0].replace("$", "").replace(",", "").strip()), 2)
                    if len(values) >= 2 else None
                )
                summary["income_year_to_date"] = (
                    round(float(values[1].replace("$", "").replace(",", "").strip()), 2)
                    if len(values) >= 2 else None
                )
            else:
                summary["income_this_period"] = None
                summary["income_year_to_date"] = None
        else:
            summary["income_this_period"] = None
            summary["income_year_to_date"] = None

        summaries.append(summary)
    return summaries


def _fidelity_holdings(text: str) -> list[dict]:
    holdings = []
    numeric_token = re.compile(
        r"not\s+applicable|n/?a|--|(?<![\w.])\$?\s*\(?-?\d[\d,]*(?:\.\d{1,4})?\)?%?",
        re.IGNORECASE,
    )
    ticker = re.compile(r"\(([A-Z][A-Z0-9.]{0,5})\)")
    for page in re.split(r"--- Page \d+ ---", text, flags=re.IGNORECASE):
        headers = list(re.finditer(r"EAI\s*\(\s*\$\s*\)\s*/\s*EY\s*\(\s*%\s*\)", page, re.IGNORECASE))
        for header_index, header in enumerate(headers):
            section_end = headers[header_index + 1].start() if header_index + 1 < len(headers) else len(page)
            section = page[header.end():section_end]
            for match in ticker.finditer(section):
                row_lines = section[match.end():].splitlines()[:3]
                row_text = "\n".join(row_lines)
                row_text = re.sub(
                    r"(?:--\s*)?\b\d+\s*-\s*day\s+yield\s*:\s*[\d.]+\s*%",
                    "",
                    row_text,
                    flags=re.IGNORECASE,
                )
                tokens = numeric_token.findall(row_text)
                if len(tokens) < 6:
                    continue
                values = []
                for token in tokens[:8]:
                    normalized = token.casefold().strip()
                    if normalized in {"not applicable", "n/a", "na", "--"}:
                        values.append(None)
                        continue
                    amount = normalized.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
                    negative = amount.startswith("(") and amount.endswith(")")
                    amount = amount.strip("()")
                    parsed = float(amount)
                    values.append(-parsed if negative else parsed)
                if len(values) < 6:
                    continue
                values.extend([None] * (8 - len(values)))
                holdings.append({
                    "beginning_market_value": values[0],
                    "quantity": values[1],
                    "price_per_unit": values[2],
                    "ending_market_value": values[3],
                    "cost_basis": values[4],
                    "unrealized_gain_loss": values[5],
                    "estimated_annual_income": values[6],
                    "estimated_yield_percent": values[7],
                })
    return holdings


def _fidelity_activity_transactions(text: str) -> list[dict]:
    transactions = []
    date_pattern = re.compile(
        r"(?<!\d)(\d{1,2}/\d{1,2}/\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})(?!\d)",
        re.IGNORECASE,
    )
    value_pattern = re.compile(
        r"(?<![\w.])(--|N/A|N\.A\.|not\s+applicable|"
        r"\(?-?\s*\$?\s*\d[\d,]*(?:\.\d+)?\)?)(?![\w.])",
        re.IGNORECASE,
    )
    action_pattern = re.compile(
        r"\b(?:YOU\s+(?:BOUGHT|SOLD)|BUY|SELL|REINVEST(?:MENT|ED)?|DIVIDEND|"
        r"INTEREST|DISTRIBUTION|REDEMPTION|TRANSFER|EXCHANGE|TAX|FEE|PURCHASE|SALE)\b",
        re.IGNORECASE,
    )
    header_pattern = re.compile(
        r"Settlement\s+Date.*?Security\s+Name.*?Symbol\s*/?\s*CUSIP.*?Description.*?"
        r"Quantity.*?Price.*?Total\s+Cost\s+Basis.*?Transaction\s+Cost.*?Amount",
        re.IGNORECASE | re.DOTALL,
    )

    for page in re.split(r"--- Page \d+ ---", text, flags=re.IGNORECASE):
        header = header_pattern.search(page)
        if not header:
            continue
        section = page[header.end():]
        date_matches = list(date_pattern.finditer(section))
        for index, date_match in enumerate(date_matches):
            end = date_matches[index + 1].start() if index + 1 < len(date_matches) else len(section)
            row_text = section[date_match.end():end].strip()
            values = list(value_pattern.finditer(row_text))
            if len(values) < 5:
                continue

            columns = values[-5:]
            leading = " ".join(row_text[:columns[0].start()].split()).strip("|")
            parsed_values = []
            for value_match in columns:
                value = value_match.group().strip()
                if value.casefold() in {"--", "n/a", "n.a.", "not applicable"}:
                    parsed_values.append(None)
                    continue
                normalized = value.replace("$", "").replace(",", "").replace(" ", "")
                negative = normalized.startswith("(") and normalized.endswith(")")
                amount = float(normalized.strip("()"))
                parsed_values.append(-amount if negative else amount)

            symbol = None
            symbol_match = re.search(r"\(([A-Z0-9][A-Z0-9.-]{1,11})\)", leading)
            if not symbol_match:
                symbol_match = re.search(r"\b([A-Z0-9]{9})\b", leading)
            if symbol_match:
                symbol = symbol_match.group(1)
                security_and_description = (
                    leading[:symbol_match.start()] + " " + leading[symbol_match.end():]
                ).strip()
            else:
                security_and_description = leading

            action = action_pattern.search(security_and_description)
            if action:
                security_name = security_and_description[:action.start()].strip(" -|")
                description = security_and_description[action.start():].strip(" -|")
                if symbol is None:
                    suffix = re.search(r"\b([A-Z][A-Z0-9.]{0,5})$", security_name)
                    if suffix and suffix.group(1) not in {
                        "COM", "INC", "CORP", "CO", "ETF", "ETN", "CL", "CLASS", "SHS", "ADR", "LP", "LLC",
                    }:
                        symbol = suffix.group(1)
                        security_name = security_name[:suffix.start()].strip(" -|")
            else:
                security_name = security_and_description
                description = ""

            try:
                settlement_date = datetime.strptime(date_match.group(1), "%m/%d/%Y")
            except ValueError:
                try:
                    settlement_date = datetime.strptime(date_match.group(1), "%m/%d/%y")
                except ValueError:
                    date_value = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", date_match.group(1), flags=re.IGNORECASE)
                    for date_format in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
                        try:
                            settlement_date = datetime.strptime(date_value, date_format)
                            break
                        except ValueError:
                            settlement_date = None
                    if settlement_date is None:
                        continue

            transactions.append({
                "settlement_date": settlement_date.date().isoformat(),
                "security_name": security_name or None,
                "symbol_cusip": symbol,
                "description": description or None,
                "quantity": parsed_values[0],
                "price": parsed_values[1],
                "total_cost_basis": parsed_values[2],
                "transaction_cost": parsed_values[3],
                "amount": parsed_values[4],
            })
    return transactions


def analyze_fidelity_statement_text(text: str) -> dict:
    """Extract only unambiguous, explicitly labeled Fidelity statement balance figures."""
    account_summaries = _fidelity_account_summaries(text)
    holdings = _fidelity_holdings(text)
    activity_transactions = _fidelity_activity_transactions(text)
    first_page = re.split(r"--- Page 2 ---", text, maxsplit=1, flags=re.IGNORECASE)[0]
    overview_labels = (
        re.compile(r"Your\s+Portfolio\s+Value\s*:?", flags=re.IGNORECASE),
        re.compile(r"Portfolio\s+Change\s+from\s+Last\s+Period\s*:?", flags=re.IGNORECASE),
        re.compile(r"Total\s+Including\s+Other\s+Holdings(?:\s*\d+)?\s*:?", flags=re.IGNORECASE),
    )
    overview_matches = []
    search_from = 0
    for label_pattern in overview_labels:
        match = label_pattern.search(first_page, search_from)
        if not match:
            overview_matches = []
            break
        overview_matches.append(match)
        search_from = match.end()

    if overview_matches:
        values_start = overview_matches[-1].end()
        summary_end = re.search(r"\bBeginning\s+Portfolio\s+Value\b", first_page[values_start:], flags=re.IGNORECASE)
        if summary_end:
            values_end = values_start + summary_end.start()
            currency_amounts = re.findall(
                r"(?<![\w.])(?P<amount>\(?-?\s*\$?\s*-?\d[\d,]*\.\d{2}\)?)",
                first_page[values_start:values_end],
            )
            if len(currency_amounts) == 3:
                parsed_amounts = []
                for amount_text in currency_amounts:
                    normalized = amount_text.replace("$", "").replace(",", "").replace(" ", "")
                    negative = normalized.startswith("(") and normalized.endswith(")")
                    amount = float(normalized.strip("()"))
                    parsed_amounts.append(round(-amount if negative else amount, 2))
                period = re.search(
                    r"\b([A-Z][a-z]+\s+\d{1,2},\s+\d{4}\s*[-–]\s*[A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b",
                    first_page,
                )
                return {
                    "document_type": "portfolio_summary",
                    "transactions": [],
                    "summary": None,
                    "portfolio_summary": {
                        "beginning_value": None,
                        "ending_value": parsed_amounts[0],
                        "reported_change": parsed_amounts[1],
                        "total_including_other_holdings": parsed_amounts[2],
                        "statement_period": period.group(1) if period else None,
                        **_fidelity_portfolio_sections(text),
                    },
                    "account_summaries": account_summaries,
                    "holdings": holdings,
                    "activity_transactions": activity_transactions,
                }

    section_summary = _fidelity_portfolio_sections(text)
    portfolio_summary = {
        "beginning_value": _unique_labeled_amount(
            text, ("Beginning Portfolio Value", "Beginning Account Value")
        ),
        "ending_value": _unique_labeled_amount(
            text, ("Ending Portfolio Value", "Ending Account Value")
        ),
        "reported_change": _unique_labeled_amount(
            text,
            (
                "Change in Portfolio Value",
                "Change in Account Value",
                "Change in Investment Value",
            ),
        ),
        "total_including_other_holdings": None,
        "statement_period": None,
        **section_summary,
    }
    if (
        any(value is not None for value in portfolio_summary.values())
        or account_summaries
        or holdings
        or activity_transactions
    ):
        return {
            "document_type": "portfolio_summary",
            "transactions": [],
            "summary": None,
            "portfolio_summary": portfolio_summary,
            "account_summaries": account_summaries,
            "holdings": holdings,
            "activity_transactions": activity_transactions,
        }
    return {
        "document_type": "statement_text",
        "transactions": [],
        "summary": None,
        "parse_diagnostics": _fidelity_overview_diagnostics(text),
        "account_summaries": account_summaries,
        "holdings": holdings,
        "activity_transactions": activity_transactions,
    }


def obfuscate_sensitive_text(text: str) -> str:
    """Mask common identifiers as defense in depth; display output is also allowlist-generated."""
    text = re.sub(
        r"(?im)^(\s*(?:name|account holder|account owner|customer name|client name|ssn|social security|"
        r"password|passcode|pwd|pin|address|mailing address|account|account number|account no\.?|acct\.?)"
        r"\s*[:#-]\s*).+$",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", "[REDACTED SSN]", text)
    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[REDACTED EMAIL]", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)", "[REDACTED PHONE]", text)
    text = re.sub(
        r"\b(?:\d{1,6}\s+[\w.'-]+\s+){0,4}(?:street|st|road|rd|avenue|ave|boulevard|blvd|drive|dr|"
        r"lane|ln|court|ct|way)\b[^\n]*",
        "[REDACTED ADDRESS]",
        text,
        flags=re.IGNORECASE,
    )
    return text


def safe_statement_display(analysis: dict) -> str:
    """Build display text only from fields explicitly allowed for view-only transfer analytics."""
    if analysis["document_type"] == "portfolio_summary":
        summary = analysis["portfolio_summary"]
        lines = ["Portfolio Summary — redacted view"]
        if summary.get("statement_period"):
            lines.append(f"Statement period | {summary['statement_period']}")
        labels = (
            ("Beginning value", "beginning_value"),
            ("Ending value", "ending_value"),
            ("Portfolio change from last period", "reported_change"),
            ("Total including other holdings", "total_including_other_holdings"),
            ("Income this period", "income_this_period"),
            ("Income year-to-date", "income_year_to_date"),
            ("Top holdings total value", "top_holdings_value"),
        )
        lines.extend(
            f"{label} | ${summary[key]:,.2f}"
            for label, key in labels
            if summary[key] is not None
        )
        if summary.get("top_holdings_percent") is not None:
            lines.append(f"Top holdings percent of portfolio | {summary['top_holdings_percent']:.0f}%")
        for index, account in enumerate(analysis.get("account_summaries", []), start=1):
            lines.append(f"Account {index} Summary")
            account_labels = (
                ("Account value", "account_value"),
                ("Change in account value", "change_in_account_value"),
                ("Beginning account value", "beginning_value"),
                ("Additions", "additions"),
                ("Subtractions", "subtractions"),
                ("Transaction costs, fees and charges", "transaction_costs"),
                ("Change in investment value", "investment_change"),
                ("Ending account value", "ending_value"),
                ("Accrued interest", "accrued_interest"),
                ("Account top holdings total", "top_holdings_value"),
                ("Account income this period", "income_this_period"),
                ("Account income year-to-date", "income_year_to_date"),
            )
            lines.extend(
                f"{label} | ${account[key]:,.2f}"
                for label, key in account_labels
                if account.get(key) is not None
            )
            if account.get("top_holdings_percent") is not None:
                lines.append(f"Account top holdings percent | {account['top_holdings_percent']:.0f}%")
        for index, holding in enumerate(analysis.get("holdings", []), start=1):
            lines.append(
                f"Holding {index} | "
                f"Beginning market value {holding['beginning_market_value']} | "
                f"Quantity {holding['quantity']} | "
                f"Price per unit {holding['price_per_unit']} | "
                f"Ending market value {holding['ending_market_value']} | "
                f"Cost basis {holding['cost_basis']} | "
                f"Unrealized gain/loss {holding['unrealized_gain_loss']} | "
                f"EAI {holding['estimated_annual_income']} | "
                f"EY percent {holding['estimated_yield_percent']}"
            )
        for transaction in analysis.get("activity_transactions", []):
            lines.append(
                f"Activity transaction | Settlement date {transaction['settlement_date']} | "
                f"Security {transaction['security_name'] or 'not available'} | "
                f"Symbol/CUSIP {transaction['symbol_cusip'] or 'not available'} | "
                f"Description {transaction['description'] or 'not available'} | "
                f"Quantity {transaction['quantity']} | Price {transaction['price']} | "
                f"Total cost basis {transaction['total_cost_basis']} | "
                f"Transaction cost {transaction['transaction_cost']} | Amount {transaction['amount']}"
            )
        return "\n".join(lines)
    if analysis["document_type"] != "transfer_activity":
        return "Statement text extracted. Personal and account details are not displayed. No analytics are available for this format."
    lines = ["Transfer Activity — redacted view", "Date | Activity | Status | Amount"]
    lines.extend(
        f"{transaction['date']} | Transfer | {transaction['status']} | ${transaction['amount']:,.2f}"
        for transaction in analysis["transactions"]
    )
    return obfuscate_sensitive_text("\n".join(lines))


def statement_structure_preview(text: str, parser_diagnostics: dict | None = None) -> str:
    """Keep useful financial labels and line shapes while replacing values/free text with placeholders."""
    output = ["Sanitized statement layout (values and identifying text replaced with placeholders)"]
    if parser_diagnostics:
        output.append(
            "Fidelity parser check: overview labels "
            f"{parser_diagnostics['labels_found']}/3; summary boundary "
            f"{'found' if parser_diagnostics['boundary_found'] else 'not found'}; "
            f"decimal amounts in overview block {parser_diagnostics['amount_count']}."
        )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("--- Page "):
            continue
        line = obfuscate_sensitive_text(line)
        line = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "<DATE>", line)
        line = re.sub(r"\$?\(?-?\d[\d,]*(?:\.\d+)?%?\)?", "<VALUE>", line)
        words = re.findall(r"[A-Za-z][A-Za-z'-]*|<REDACTED[^>]*>|\[REDACTED\]|[|]+", line)
        tokens = [
            word.upper() if word.casefold() in STRUCTURE_LABELS
            else word if word.startswith(("<REDACTED", "[REDACTED"))
            else "|" if word == "||"
            else "<TEXT>"
            for word in words
        ]
        shape = " ".join(tokens)
        shape = re.sub(r"(?:<TEXT> ){2,}", "<TEXT> ", shape)
        shape = re.sub(r" {2,}", " ", shape).strip()
        if not shape or shape in output[-1:]:
            continue
        output.append(shape)
        if len(output) >= MAX_STRUCTURE_LINES:
            output.append("[Preview limited to 120 lines]")
            break
    return "\n".join(output)


def read_statement_pdf(filename: str, content: bytes, broker_name: str = "E*TRADE") -> dict:
    if not content:
        raise StatementPdfError("The uploaded PDF is empty.")
    if len(content) > MAX_PDF_BYTES:
        raise StatementPdfError("The PDF exceeds the 20 MB upload limit.")
    if not content.startswith(b"%PDF-"):
        raise StatementPdfError("The uploaded file is not a valid PDF.")

    try:
        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise StatementPdfError("Password-protected PDFs are not supported.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise StatementPdfError("The PDF exceeds the 500-page limit.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except StatementPdfError:
        raise
    except (PdfReadError, OSError) as exc:
        raise StatementPdfError("The PDF could not be read. Please check that it is not damaged.") from exc

    text = "\n\n".join(f"--- Page {index} ---\n{page.strip()}" for index, page in enumerate(pages, 1)
                       if page.strip())
    if not text.strip():
        raise StatementPdfError(
            "No selectable text was found. Scanned/image-only PDFs are not supported yet; "
            "upload a text-based PDF statement."
        )
    if len(text) > MAX_EXTRACTED_TEXT_CHARS:
        raise StatementPdfError("The PDF contains too much text to keep in memory (2 million characters maximum).")

    transfer_analysis = analyze_statement_text(obfuscate_sensitive_text(text))
    if broker_name.casefold() == "fidelity":
        analysis = analyze_fidelity_statement_text(text)
        if analysis["document_type"] == "portfolio_summary":
            display_text = safe_statement_display(analysis)
        elif transfer_analysis["document_type"] == "transfer_activity":
            analysis = transfer_analysis
            display_text = safe_statement_display(analysis)
        else:
            analysis["document_type"] = "statement_structure"
            diagnostics = analysis.pop("parse_diagnostics", None)
            display_text = statement_structure_preview(text, diagnostics)
    elif transfer_analysis["document_type"] == "transfer_activity":
        analysis = transfer_analysis
        display_text = safe_statement_display(analysis)
    else:
        analysis = transfer_analysis
        analysis["document_type"] = "statement_structure"
        display_text = statement_structure_preview(text)
    return {
        "filename": f"{broker_name} statement",
        "page_count": len(pages),
        "text": display_text,
        "character_count": len(display_text),
        "imported_at": datetime.now(timezone.utc).isoformat(),
        **analysis,
    }
