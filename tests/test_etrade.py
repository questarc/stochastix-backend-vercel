import pytest

from app import deps
from app.etrade import (
    StatementPdfError,
    analyze_fidelity_statement_text,
    analyze_statement_text,
    obfuscate_sensitive_text,
    read_statement_pdf,
    safe_statement_display,
    statement_structure_preview,
)
from app.stores import InMemoryEtradeStatementStore


def test_statement_pdf_text_extraction_and_filename_sanitizing(monkeypatch):
    class Page:
        def extract_text(self):
            return "E*TRADE Account Statement\nTotal Account Value $123.45"

    class Reader:
        is_encrypted = False
        pages = [Page()]

        def __init__(self, _file, strict):
            assert strict is False

    monkeypatch.setattr("app.etrade.PdfReader", Reader)
    statement = read_statement_pdf(r"C:\private\statement.pdf", b"%PDF-test")

    assert statement["filename"] == "E*TRADE statement"
    assert statement["page_count"] == 1
    assert "TOTAL ACCOUNT VALUE" in statement["text"]
    assert "$123.45" not in statement["text"]
    assert statement["character_count"] == len(statement["text"])


def test_statement_pdf_rejects_non_pdf_and_scanned_pdf(monkeypatch):
    with pytest.raises(StatementPdfError, match="not a valid PDF"):
        read_statement_pdf("statement.pdf", b"not a pdf")

    class Page:
        def extract_text(self):
            return ""

    class Reader:
        is_encrypted = False
        pages = [Page()]

        def __init__(self, _file, strict):
            pass

    monkeypatch.setattr("app.etrade.PdfReader", Reader)
    with pytest.raises(StatementPdfError, match="Scanned/image-only PDFs"):
        read_statement_pdf("scanned.pdf", b"%PDF-test")


def test_transfer_activity_extracts_cash_flow_without_account_details():
    text = (
        "Transfer Activity\n"
        "\ue000Scheduled \ue001Complete \ue002In process\n"
        "\ue000 10/13/2026 Transfer money $75,000.00 private bank ...1234 Savings ...9969 Once\n"
        "\ue001 10/05/2026 Transfer money $25,000.00 private bank ...1234 Savings ...9969 Once\n"
        "\ue001 09/28/2026 Transfer money $100.00 private bank ...1234 Savings ...9969 Once\n"
    )
    analysis = analyze_statement_text(text)

    assert analysis["document_type"] == "transfer_activity"
    assert analysis["summary"] == {
        "transaction_count": 3,
        "total_amount": 100100.0,
        "completed_amount": 25100.0,
        "scheduled_amount": 75000.0,
    }
    assert [row["status"] for row in analysis["transactions"]] == ["Scheduled", "Complete", "Complete"]
    assert "private bank" not in str(analysis)
    preview = safe_statement_display(analysis)
    assert "private bank" not in preview and "1234" not in preview and "Savings" not in preview
    assert "2026-10-13" in preview and "$75,000.00" in preview


def test_obfuscation_masks_common_sensitive_fields():
    private_text = (
        "Name: Jane Doe\n"
        "SSN: 123-45-6789\n"
        "Password: SecretPass123\n"
        "Address: 123 Main Street, Springfield, NY 10001\n"
        "Email: jane@example.com\n"
        "Phone: (212) 555-0199\n"
    )
    redacted = obfuscate_sensitive_text(private_text)

    for private_value in (
        "Jane Doe", "123-45-6789", "SecretPass123", "123 Main Street",
        "jane@example.com", "(212) 555-0199",
    ):
        assert private_value not in redacted
    assert redacted.count("[REDACTED]") >= 4
    assert "[REDACTED EMAIL]" in redacted
    assert "[REDACTED PHONE]" in redacted


def test_unknown_statement_preview_keeps_layout_without_personal_values():
    text = (
        "--- Page 1 ---\n"
        "Account Summary\n"
        "Name: Jane Doe\n"
        "SSN: 123-45-6789\n"
        "123 Main Street, Springfield, NY 10001\n"
        "Portfolio Holdings\n"
        "Symbol Description Quantity Price Market Value\n"
        "AAPL Apple Inc 25 $200.00 $5,000.00\n"
        "Account Number: 123456789\n"
    )
    preview = statement_structure_preview(text)

    assert "ACCOUNT SUMMARY" in preview
    assert "PORTFOLIO HOLDINGS" in preview
    assert "SYMBOL DESCRIPTION QUANTITY PRICE MARKET VALUE" in preview
    for private_value in (
        "Jane", "Doe", "123-45-6789", "Main Street", "Springfield", "123456789", "AAPL", "Apple", "200.00",
    ):
        assert private_value not in preview


def test_fidelity_summary_extracts_explicit_balance_labels_only():
    text = (
        "Portfolio Summary\n"
        "Beginning Portfolio Value $10,000.00\n"
        "Change in Investment Value $125.50\n"
        "Ending Portfolio Value $10,125.50\n"
        "Name: Jane Doe\n"
        "Account Number: 123456789\n"
    )
    analysis = analyze_fidelity_statement_text(text)

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["portfolio_summary"] == {
        "beginning_value": 10000.0,
        "ending_value": 10125.5,
        "reported_change": 125.5,
        "total_including_other_holdings": None,
        "statement_period": None,
        "income_this_period": None,
        "income_year_to_date": None,
        "top_holdings_value": None,
        "top_holdings_percent": None,
    }
    preview = safe_statement_display(analysis)
    assert "Jane Doe" not in preview
    assert "123456789" not in preview
    assert "$10,000.00" in preview


def test_fidelity_overview_maps_amounts_after_ordered_labels():
    analysis = analyze_fidelity_statement_text(
        "INVESTMENT REPORT\n"
        "September 1, 2026 - September 30, 2026\n"
        "Your Portfolio Value:\n"
        "Portfolio Change from Last Period:\n"
        "Total Including Other Holdings1:\n"
        "$123,456.78 $2,345.67 $124,000.00\n"
        "This Period Year-to-Date Beginning Portfolio Value $120,000.00 $110,000.00\n"
        "--- Page 2 ---\n"
        "Account Number: private-value\n"
    )

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["portfolio_summary"] == {
        "beginning_value": None,
        "ending_value": 123456.78,
        "reported_change": 2345.67,
        "total_including_other_holdings": 124000.0,
        "statement_period": "September 1, 2026 - September 30, 2026",
        "income_this_period": None,
        "income_year_to_date": None,
        "top_holdings_value": None,
        "top_holdings_percent": None,
    }
    preview = safe_statement_display(analysis)
    assert "private-value" not in preview
    assert "Portfolio change from last period | $2,345.67" in preview


def test_fidelity_portfolio_summary_extracts_income_and_top_holdings_totals():
    analysis = analyze_fidelity_statement_text(
        "Portfolio Summary\n"
        "Accounts Included in This Report\n"
        "Income Summary\n"
        "This Period\n"
        "Year-to-Date\n"
        "Taxable\n"
        "Dividends\n"
        "$100.00\n"
        "100.00\n"
        "$1,000.00\n"
        "1,000.00\n"
        "Total\n"
        "$100.00\n"
        "$1,000.00\n"
        "Top Holdings\n"
        "Description\n"
        "Value\n"
        "Percent of Portfolio\n"
        "Fund One\n"
        "Company Two\n"
        "$80,000\n"
        "20,000\n"
        "80%\n"
        "20\n"
        "Total\n"
        "$100,000\n"
        "100%\n"
    )

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["portfolio_summary"] == {
        "beginning_value": None,
        "ending_value": None,
        "reported_change": None,
        "total_including_other_holdings": None,
        "statement_period": None,
        "income_this_period": 100.0,
        "income_year_to_date": 1000.0,
        "top_holdings_value": 100000.0,
        "top_holdings_percent": 100.0,
    }
    assert analysis["account_summaries"] == []


def test_fidelity_account_summary_extracts_numeric_metrics_without_identifiers():
    analysis = analyze_fidelity_statement_text(
        "Account # PRIVATE-ID\n"
        "Account Summary\n"
        "Account Holdings\n"
        "87% Stocks ($54,567)\n"
        "9% Exchange Traded Products ($5,776)\n"
        "4% Core Account ($2,554)\n"
        "Account Value: $62,899.28\n"
        "Change in Account Value $114.36\n"
        "This Period Year-to-Date\n"
        "Beginning Account Value $63,013.64 $20,263.63\n"
        "Additions-40,000.00\n"
        "Subtractions-0.97-5.24\n"
        "Transaction Costs, Fees & Charges-0.97-4.52\n"
        "Change in Investment Value *-113.39 2,640.89\n"
        "Ending Account Value $62,899.28 $62,899.28\n"
        "Accrued Interest (AI) 0.00\n"
        "Ending Account Value Incl. AI $62,899.28\n"
        "Top Holdings\n"
        "Description Value Percent of Account\n"
        "Company One $19,117 30%\n"
        "Company Two 13,848 22\n"
        "Total $39,780 63%\n"
        "Income Summary\n"
        "This Period Year-to-Date\n"
        "Taxable $45.15 $451.19\n"
        "Dividends 45.15 451.19\n"
        "Total $45.15 $451.19\n"
    )

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["account_summaries"] == [{
        "account_value": 62899.28,
        "change_in_account_value": 114.36,
        "beginning_value": 63013.64,
        "additions": -40000.0,
        "subtractions": -0.97,
        "transaction_costs": -0.97,
        "investment_change": -113.39,
        "ending_value": 62899.28,
        "accrued_interest": 0.0,
        "top_holdings_value": 39780.0,
        "top_holdings_percent": 63.0,
        "income_this_period": 45.15,
        "income_year_to_date": 451.19,
    }]


def test_fidelity_holdings_extracts_anonymous_rows_from_reported_columns():
    analysis = analyze_fidelity_statement_text(
        "Holdings\n"
        "Description\nBeginning Market Value\nQuantity\nPrice Per Unit\n"
        "Ending Market Value\nTotal Cost Basis\nUnrealized Gain/Loss\nEAI ($) / EY (%)\n"
        "FIDELITY GOVERNMENT MONEY MARKET (SPAXX)-- 7-day yield: 3.48%\n"
        "$32,764.81 2,554.980 $1.0000 $2,554.98 not applicable not applicable $87.66\n"
        "3.430%\n"
        "STATE STREET SPDR S&P 500 ETF UNITS (SPY)\n"
        "$1,433.61 3.869 $762.6300 $2,950.61 $2,779.32 $171.29 $29.34\n"
        "0.990%\n"
        "VANGUARD INDEX FUNDS S&P 500 ETF USD (VOO)\n"
        "666.12 2.945 700.8600 2,064.03 1,999.91 64.12 21.88\n"
        "1.060\n"
    )

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["holdings"] == [
        {
            "beginning_market_value": 32764.81,
            "quantity": 2554.98,
            "price_per_unit": 1.0,
            "ending_market_value": 2554.98,
            "cost_basis": None,
            "unrealized_gain_loss": None,
            "estimated_annual_income": 87.66,
            "estimated_yield_percent": 3.43,
        },
        {
            "beginning_market_value": 1433.61,
            "quantity": 3.869,
            "price_per_unit": 762.63,
            "ending_market_value": 2950.61,
            "cost_basis": 2779.32,
            "unrealized_gain_loss": 171.29,
            "estimated_annual_income": 29.34,
            "estimated_yield_percent": 0.99,
        },
        {
            "beginning_market_value": 666.12,
            "quantity": 2.945,
            "price_per_unit": 700.86,
            "ending_market_value": 2064.03,
            "cost_basis": 1999.91,
            "unrealized_gain_loss": 64.12,
            "estimated_annual_income": 21.88,
            "estimated_yield_percent": 1.06,
        },
    ]
    display = safe_statement_display(analysis)
    assert "SPY" not in display and "VOO" not in display
    assert "STATE STREET" not in display and "VANGUARD" not in display


def test_fidelity_activity_extracts_and_lists_requested_transaction_columns():
    analysis = analyze_fidelity_statement_text(
        "Activity\n"
        "Settlement Date\n"
        "Security Name\n"
        "Symbol/CUSIP\n"
        "Description\n"
        "Quantity\n"
        "Price\n"
        "Total Cost Basis\n"
        "Transaction Cost\n"
        "Amount\n"
        "10/05/2026 COSTCO WHOLESALE CORP COM (COST) YOU BOUGHT "
        "1.000 $100.00 $90.00 $0.01 $100.01\n"
        "10/06/2026 APPLE INC AAPL DIVIDEND "
        "2 $1.00 -- -- $2.00\n"
        "Account Number: private-account\n"
    )

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["activity_transactions"] == [
        {
            "settlement_date": "2026-10-05",
            "security_name": "COSTCO WHOLESALE CORP COM",
            "symbol_cusip": "COST",
            "description": "YOU BOUGHT",
            "quantity": 1.0,
            "price": 100.0,
            "total_cost_basis": 90.0,
            "transaction_cost": 0.01,
            "amount": 100.01,
        },
        {
            "settlement_date": "2026-10-06",
            "security_name": "APPLE INC",
            "symbol_cusip": "AAPL",
            "description": "DIVIDEND",
            "quantity": 2.0,
            "price": 1.0,
            "total_cost_basis": None,
            "transaction_cost": None,
            "amount": 2.0,
        },
    ]
    display = safe_statement_display(analysis)
    assert "COSTCO WHOLESALE CORP COM" in display
    assert "AAPL" in display
    assert "private-account" not in display


def test_fidelity_overview_parses_extracted_first_page_line_breaks():
    analysis = analyze_fidelity_statement_text(
        "INVESTMENT REPORT\n"
        "September 1, 2026 - September 30, 2026\n"
        "Your Portfolio Value:\n"
        "Portfolio Change from Last Period:\n"
        "Total Including Other Holdings1:\n"
        "$111,111.11\n"
        "$2,222.22\n"
        "$113,333.33\n"
        "This Period\n"
        "Year-to-Date\n"
        "Beginning Portfolio Value\n"
        "$100,000.00\n"
        "Ending Portfolio Value **\n"
        "$111,111.11\n"
        "--- Page 2 ---\n"
        "Portfolio Summary\n"
    )

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["portfolio_summary"] == {
        "beginning_value": None,
        "ending_value": 111111.11,
        "reported_change": 2222.22,
        "total_including_other_holdings": 113333.33,
        "statement_period": "September 1, 2026 - September 30, 2026",
        "income_this_period": None,
        "income_year_to_date": None,
        "top_holdings_value": None,
        "top_holdings_percent": None,
    }


def test_fidelity_overview_parses_amounts_even_when_pdf_drops_currency_symbols():
    analysis = analyze_fidelity_statement_text(
        "INVESTMENT REPORT\n"
        "September 1, 2026 - September 30, 2026\n"
        "Your Portfolio Value:\n"
        "Portfolio Change from Last Period:\n"
        "Total Including Other Holdings1:\n"
        "$111,111.11\n"
        "2,222.22\n"
        "113,333.33\n"
        "This Period\n"
        "Year-to-Date\n"
        "Beginning Portfolio Value\n"
        "$100,000.00\n"
        "--- Page 2 ---\n"
    )

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["portfolio_summary"]["ending_value"] == 111111.11
    assert analysis["portfolio_summary"]["reported_change"] == 2222.22
    assert analysis["portfolio_summary"]["total_including_other_holdings"] == 113333.33


def test_fidelity_overview_does_not_depend_on_this_period_heading_after_values():
    analysis = analyze_fidelity_statement_text(
        "Your Portfolio Value: Portfolio Change from Last Period: "
        "Total Including Other Holdings1: "
        "This Period Year-to-Date $123,456.78 $2,345.67 $124,000.00 "
        "Beginning Portfolio Value $120,000.00 $110,000.00"
    )

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["portfolio_summary"]["ending_value"] == 123456.78
    assert analysis["portfolio_summary"]["reported_change"] == 2345.67
    assert analysis["portfolio_summary"]["total_including_other_holdings"] == 124000.0


def test_fidelity_overview_takes_precedence_over_transfer_text(monkeypatch):
    class Page:
        def extract_text(self):
            return (
                "INVESTMENT REPORT September 1, 2026 - September 30, 2026\n"
                "Your Portfolio Value: Portfolio Change from Last Period: "
                "Total Including Other Holdings1: $123,456.78 $2,345.67 $124,000.00\n"
                "This Period Year-to-Date Beginning Portfolio Value\n"
                "Transfer Activity\n"
            )

    class Reader:
        is_encrypted = False
        pages = [Page()]

        def __init__(self, _file, strict):
            pass

    monkeypatch.setattr("app.etrade.PdfReader", Reader)
    statement = read_statement_pdf("private.pdf", b"%PDF-test", "Fidelity")

    assert statement["document_type"] == "portfolio_summary"
    assert statement["portfolio_summary"]["ending_value"] == 123456.78
    assert statement["portfolio_summary"]["reported_change"] == 2345.67
    assert statement["portfolio_summary"]["total_including_other_holdings"] == 124000.0
    assert "Portfolio Summary — redacted view" in statement["text"]


def test_fidelity_summary_omits_balance_labels_with_ambiguous_values():
    analysis = analyze_fidelity_statement_text(
        "Ending Account Value $10,000.00\nEnding Account Value $20,000.00"
    )

    assert analysis["document_type"] == "statement_text"
    assert analysis["parse_diagnostics"] == {
        "labels_found": 0,
        "boundary_found": False,
        "amount_count": 0,
    }


def test_fidelity_summary_reads_amounts_wrapped_to_next_line():
    analysis = analyze_fidelity_statement_text(
        "Beginning Portfolio Value\n$10,000.00\n"
        "Ending Portfolio Value\n$10,125.50\n"
    )

    assert analysis["document_type"] == "portfolio_summary"
    assert analysis["portfolio_summary"] == {
        "beginning_value": 10000.0,
        "ending_value": 10125.5,
        "reported_change": None,
        "total_including_other_holdings": None,
        "statement_period": None,
        "income_this_period": None,
        "income_year_to_date": None,
        "top_holdings_value": None,
        "top_holdings_percent": None,
    }


def test_fidelity_summary_ignores_multiple_values_after_same_label():
    analysis = analyze_fidelity_statement_text(
        "Ending Portfolio Value $10,125.50 $20,000.00"
    )

    assert analysis["document_type"] == "statement_text"
    assert analysis["parse_diagnostics"]["labels_found"] == 0


def test_unknown_fidelity_pdf_includes_only_safe_parser_diagnostics(monkeypatch):
    class Page:
        def extract_text(self):
            return (
                "Your Portfolio Value:\n"
                "Portfolio Change from Last Period:\n"
                "Total Including Other Holdings1:\n"
                "This Period\n"
                "Beginning Portfolio Value\n"
                "Private Account Label $987,654.32\n"
            )

    class Reader:
        is_encrypted = False
        pages = [Page()]

        def __init__(self, _file, strict):
            pass

    monkeypatch.setattr("app.etrade.PdfReader", Reader)
    statement = read_statement_pdf("private.pdf", b"%PDF-test", "Fidelity")

    assert statement["document_type"] == "statement_structure"
    assert "Fidelity parser check: overview labels 3/3" in statement["text"]
    assert "summary boundary found" in statement["text"]
    assert "decimal amounts in overview block 0" in statement["text"]
    assert "987,654.32" not in statement["text"]


def test_fidelity_pdf_returns_safe_portfolio_summary(monkeypatch):
    class Page:
        def extract_text(self):
            return (
                "Beginning Account Value $10,000.00\n"
                "Ending Account Value $10,125.50\n"
                "Name: Jane Doe\n"
                "Account Number: 123456789\n"
            )

    class Reader:
        is_encrypted = False
        pages = [Page()]

        def __init__(self, _file, strict):
            pass

    monkeypatch.setattr("app.etrade.PdfReader", Reader)
    statement = read_statement_pdf("private.pdf", b"%PDF-test", "Fidelity")

    assert statement["document_type"] == "portfolio_summary"
    assert statement["portfolio_summary"] == {
        "beginning_value": 10000.0,
        "ending_value": 10125.5,
        "reported_change": None,
        "total_including_other_holdings": None,
        "statement_period": None,
        "income_this_period": None,
        "income_year_to_date": None,
        "top_holdings_value": None,
        "top_holdings_percent": None,
    }
    assert "Jane Doe" not in statement["text"]
    assert "123456789" not in statement["text"]


def test_statement_store_retains_only_fidelity_summary_allowlist():
    store = InMemoryEtradeStatementStore()
    store.set(
        "browser-session-0123456789",
        {
            "filename": "Fidelity statement",
            "page_count": 1,
            "text": "Portfolio Summary — redacted view\nEnding value | $10,125.50",
            "character_count": 54,
            "imported_at": "2026-10-31T12:00:00+00:00",
            "document_type": "portfolio_summary",
            "transactions": [],
            "summary": None,
            "account_summaries": [{
                "account_value": 62899.28,
                "change_in_account_value": 114.36,
                "beginning_value": 63013.64,
                "additions": -40000.0,
                "subtractions": -0.97,
                "transaction_costs": -0.97,
                "investment_change": -113.39,
                "ending_value": 62899.28,
                "accrued_interest": 0.0,
                "income_this_period": 45.15,
                "income_year_to_date": 451.19,
                "top_holdings_value": 39780.0,
                "top_holdings_percent": 63.0,
                "account_number": "123456789",
                "account_name": "Jane Doe",
            }],
            "holdings": [{
                "beginning_market_value": 1433.61,
                "quantity": 3.869,
                "price_per_unit": 762.63,
                "ending_market_value": 2950.61,
                "cost_basis": 2779.32,
                "unrealized_gain_loss": 171.29,
                "estimated_annual_income": 29.34,
                "estimated_yield_percent": 0.99,
                "ticker": "SPY",
                "description": "Sensitive company",
            }],
            "activity_transactions": [{
                "settlement_date": "2026-10-05",
                "security_name": "COSTCO WHOLESALE CORP COM",
                "symbol_cusip": "COST",
                "description": "YOU BOUGHT",
                "quantity": 1.0,
                "price": 100.0,
                "total_cost_basis": 90.0,
                "transaction_cost": 0.01,
                "amount": 100.01,
                "account_number": "private-account",
            }],
            "portfolio_summary": {
                "beginning_value": 10000.0,
                "ending_value": 10125.5,
                "reported_change": None,
                "total_including_other_holdings": 12000.0,
                "statement_period": "October 1, 2026 - October 31, 2026",
                "income_this_period": 100.0,
                "income_year_to_date": 1000.0,
                "top_holdings_value": 11000.0,
                "top_holdings_percent": 73.0,
                "account_number": "123456789",
            },
        },
    )

    stored = store.get("browser-session-0123456789")
    assert stored["portfolio_summary"] == {
        "beginning_value": 10000.0,
        "ending_value": 10125.5,
        "reported_change": None,
        "total_including_other_holdings": 12000.0,
        "statement_period": "October 1, 2026 - October 31, 2026",
        "income_this_period": 100.0,
        "income_year_to_date": 1000.0,
        "top_holdings_value": 11000.0,
        "top_holdings_percent": 73.0,
    }
    assert stored["text"] == ""
    assert stored["account_summaries"] == [{
        "account_value": 62899.28,
        "change_in_account_value": 114.36,
        "beginning_value": 63013.64,
        "additions": -40000.0,
        "subtractions": -0.97,
        "transaction_costs": -0.97,
        "investment_change": -113.39,
        "ending_value": 62899.28,
        "accrued_interest": 0.0,
        "income_this_period": 45.15,
        "income_year_to_date": 451.19,
        "top_holdings_value": 39780.0,
        "top_holdings_percent": 63.0,
    }]
    assert stored["holdings"] == [{
        "beginning_market_value": 1433.61,
        "quantity": 3.869,
        "price_per_unit": 762.63,
        "ending_market_value": 2950.61,
        "cost_basis": 2779.32,
        "unrealized_gain_loss": 171.29,
        "estimated_annual_income": 29.34,
        "estimated_yield_percent": 0.99,
    }]
    assert stored["activity_transactions"] == [{
        "settlement_date": "2026-10-05",
        "security_name": "COSTCO WHOLESALE CORP COM",
        "symbol_cusip": "COST",
        "description": "YOU BOUGHT",
        "quantity": 1.0,
        "price": 100.0,
        "total_cost_basis": 90.0,
        "transaction_cost": 0.01,
        "amount": 100.01,
    }]


def test_etrade_statement_upload_persists_in_memory_until_cleared(client, monkeypatch):
    deps.get_etrade_statement_store.cache_clear()
    session_a = {"X-Statement-Session": "browser-session-0123456789"}
    session_b = {"X-Statement-Session": "browser-session-9876543210"}
    parsed = {
        "filename": "statement.pdf",
        "page_count": 1,
        "text": "Account Value $123.45",
        "character_count": 21,
        "imported_at": "2026-10-31T12:00:00+00:00",
        "document_type": "statement_text",
        "transactions": [],
        "summary": None,
        "account_summaries": [],
        "holdings": [],
        "activity_transactions": [],
        "portfolio_summary": None,
    }
    monkeypatch.setattr("app.routers.etrade.read_statement_pdf", lambda filename, content: parsed)

    assert client.get("/api/etrade/statement").status_code == 422
    assert client.get("/api/etrade/statement", headers=session_a).json() is None
    response = client.post(
        "/api/etrade/statement/upload",
        files={"file": ("statement.pdf", b"%PDF-test", "application/pdf")},
        headers=session_a,
    )
    assert response.status_code == 200 and response.json() == parsed
    stored = client.get("/api/etrade/statement", headers=session_a).json()
    assert stored["text"] == ""
    assert stored["summary"] == parsed["summary"]
    assert client.get("/api/etrade/statement", headers=session_b).json() is None
    assert client.delete("/api/etrade/statement", headers=session_a).json() == {"cleared": True}
    assert client.get("/api/etrade/statement", headers=session_a).json() is None
    assert client.get("/api/etrade/statement", headers=session_b).json() is None


def test_fidelity_statement_store_is_separate_from_etrade(client, monkeypatch):
    session = {"X-Statement-Session": "shared-browser-session-012345"}
    parsed = {
        "filename": "Fidelity statement",
        "page_count": 1,
        "text": "Transfer Activity — redacted view",
        "character_count": 32,
        "imported_at": "2026-10-31T12:00:00+00:00",
        "document_type": "transfer_activity",
        "transactions": [],
        "summary": None,
        "portfolio_summary": None,
    }
    monkeypatch.setattr("app.routers.fidelity.read_statement_pdf", lambda filename, content, broker: parsed)

    response = client.post(
        "/api/fidelity/statement/upload",
        files={"file": ("statement.pdf", b"%PDF-test", "application/pdf")},
        headers=session,
    )
    assert response.status_code == 200 and response.json()["filename"] == "Fidelity statement"
    stored = client.get("/api/fidelity/statement", headers=session).json()
    assert stored["filename"] == "Fidelity statement" and stored["text"] == ""
    assert client.get("/api/etrade/statement", headers=session).json() is None


def test_fidelity_positions_csv_upload_is_session_scoped_and_excludes_account_details(client):
    session_a = {"X-Statement-Session": "fidelity-browser-session-012345"}
    session_b = {"X-Statement-Session": "fidelity-browser-session-987654"}
    csv_content = (
        "Account number,Account name,Symbol,Description,Quantity,Last price,Last price change,"
        "Current value,Today's gain/loss dollar,Today's gain/loss percent,Total gain/loss dollar,"
        "Total gain/loss percent,Percent of account,Cost basis total,Average cost basis,Type\n"
        "private-account,Private Name,GLD,SPDR GOLD TR GOLD SHS,2,$355.78,-$12.79,$711.56,"
        "-$25.58,-3.48%,-$111.41,-13.54%,1.12%,$822.97,$411.49,Cash\n"
    )

    response = client.post(
        "/api/fidelity/positions/upload",
        files={"file": ("positions.csv", csv_content.encode(), "text/csv")},
        headers=session_a,
    )

    assert response.status_code == 200
    assert response.json()[0]["symbol"] == "GLD"
    assert "private-account" not in response.text and "Private Name" not in response.text
    assert client.get("/api/fidelity/positions", headers=session_a).json() == response.json()
    assert client.get("/api/fidelity/positions", headers=session_b).json() is None
    assert client.delete("/api/fidelity/positions", headers=session_a).json() == {"cleared": True}
    assert client.get("/api/fidelity/positions", headers=session_a).json() is None
