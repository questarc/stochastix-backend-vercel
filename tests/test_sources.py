import pytest

from app.services import find_account_ids, load_account_portfolio, resolve_model_id
from app.sources import load_sources
from app.sources.positions_agent import parse_fidelity_positions_csv, parse_positions, parse_number

CONFIG = {
    "ticker_asset_class": {"VXUS": "Intl Equity", "BND": "US Bonds"},
    "sec_type_asset_class": {"Annuity": "US Bonds", "Money Market": "Cash"},
    "default_asset_class": "US Equity",
}
HEADER = "Ticker,Cusip,Sec Type,Qty,Custodian Quantity,PM Price,Market Value\n"


@pytest.fixture
def sources():
    return load_sources("config/sources.json")


@pytest.mark.parametrize("raw, expected", [("$1,234.50", 1234.5), ("+12", 12), ("(123)", -123), ("", 0), (None, 0)])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_blank_ticker_uses_cusip():
    text = HEADER + ',12345A678,Annuity,1,,,"$120,000.00"\n'
    [p] = parse_positions(text, "A1", CONFIG)
    assert p.ticker == "12345A678" and p.asset_class == "US Bonds" and p.price == 120_000


def test_missing_price_uses_market_value_over_quantity():
    text = HEADER + 'VTI,922908769,ETF,10,,,"$3,000.00"\n'
    [p] = parse_positions(text, "A1", CONFIG)
    assert p.price == 300 and p.asset_class == "US Equity"


def test_blank_qty_uses_custodian_quantity_and_skips_blank_rows():
    text = HEADER + "SPAXX,,Money Market,,50,$1.00,$50.00\n,,,,,,\nVXUS,,ETF,0,,$65,$0\n"
    positions = parse_positions(text, "A1", CONFIG)
    assert [(p.ticker, p.quantity, p.asset_class) for p in positions] == [("SPAXX", 50, "Cash")]


def test_ticker_override_beats_sec_type():
    text = HEADER + "BND,,ETF,10,,$73,$730\n"
    [p] = parse_positions(text, "A1", CONFIG)
    assert p.asset_class == "US Bonds"


def test_fidelity_positions_export_loads_all_accounts_into_selected_account():
    text = (
        "Account number,Account name,Symbol,Description,Quantity,Last price,Last price change,"
        "Current value,Today's gain/loss dollar,Today's gain/loss percent,Total gain/loss dollar,"
        "Total gain/loss percent,Percent of account,Cost basis total,Average cost basis,Type\n"
        "X83851048,Individual - TOD,SPAXX**,HELD IN MONEY MARKET,,,,$335.70,,,,,0.53%,,,Cash\n"
        "X83851048,Individual - TOD,GLD,SPDR GOLD TR GOLD SHS,2,$355.78,-$12.79,$711.56,"
        "-$25.58,-3.48%,-$111.41,-13.54%,1.12%,$822.97,$411.49,Cash\n"
        'X90712034,Individual - TOD,MSFT,MICROSOFT CORP,14,$521.91,-$8.23,"$7,306.74",'
        '-$115.22,-1.56%,+$436.20,+6.34%,2.65%,"$6,870.54",$490.75,Cash\n'
        'X90712034,Individual - TOD,Pending activity,,,,,"-$11,831.15",,,,,,\n'
    )

    positions = parse_positions(text, "SELECTED", CONFIG)

    assert [(p.ticker, p.asset_class, p.quantity, p.price, p.market_value) for p in positions] == [
        ("SPAXX", "Cash", 335.7, 1.0, 335.7),
        ("GLD", "US Equity", 2.0, 355.78, 711.56),
        ("MSFT", "US Equity", 14.0, 521.91, 7306.74),
    ]
    assert {position.account_id for position in positions} == {"SELECTED"}


def test_fidelity_positions_import_handles_unquoted_thousands_separators():
    text = (
        "Account number,Account name,Symbol,Description,Quantity,Last price,Last price change,"
        "Current value,Today's gain/loss dollar,Today's gain/loss percent,Total gain/loss dollar,"
        "Total gain/loss percent,Percent of account,Cost basis total,Average cost basis,Type\n"
        'X90712034,Individual - TOD,MSFT,MICROSOFT CORP,14,$521.91,-$8.23,$7,306.74,'
        '-$115.22,-1.56%,+$436.20,+6.34%,2.65%,$6,870.54,$490.75,Cash\n'
        'X90712034,Individual - TOD,AMZN,AMAZON.COM INC,5,$240.35,-$3.44,$1,201.75,'
        '-$17.20,-1.42%,-$41.15,-3.32%,0.44%,$1,242.90,$248.58,Cash\n'
    )

    positions = parse_positions(text.encode("utf-8"), "SELECTED", CONFIG)

    assert [(p.ticker, p.quantity, p.price, p.market_value) for p in positions] == [
        ("MSFT", 14.0, 521.91, 7306.74),
        ("AMZN", 5.0, 240.35, 1201.75),
    ]


def test_fidelity_positions_csv_parser_keeps_display_metrics_not_account_details():
    text = (
        "Account number,Account name,Symbol,Description,Quantity,Last price,Last price change,"
        "Current value,Today's gain/loss dollar,Today's gain/loss percent,Total gain/loss dollar,"
        "Total gain/loss percent,Percent of account,Cost basis total,Average cost basis,Type\n"
        "X83851048,Individual - TOD,SPAXX**,HELD IN MONEY MARKET,,,,$335.70,,,,,0.53%,,,Cash\n"
        "X90712034,Individual - TOD,GLD,SPDR GOLD TR GOLD SHS,2,$355.78,-$12.79,$711.56,"
        "-$25.58,-3.48%,-$111.41,-13.54%,1.12%,$822.97,$411.49,Cash\n"
        "X90712034,Individual - TOD,Pending activity,,,,,-$11831.15,,,,,,\n"
    )

    positions = parse_fidelity_positions_csv(text.encode("utf-8"))

    assert positions == [
        {
            "symbol": "SPAXX",
            "description": "HELD IN MONEY MARKET",
            "quantity": 335.7,
            "last_price": 1.0,
            "last_price_change": None,
            "current_value": 335.7,
            "todays_gain_loss_dollar": None,
            "todays_gain_loss_percent": None,
            "total_gain_loss_dollar": None,
            "total_gain_loss_percent": None,
            "percent_of_account": 0.53,
            "cost_basis_total": None,
            "average_cost_basis": None,
            "type": "Cash",
        },
        {
            "symbol": "GLD",
            "description": "SPDR GOLD TR GOLD SHS",
            "quantity": 2.0,
            "last_price": 355.78,
            "last_price_change": -12.79,
            "current_value": 711.56,
            "todays_gain_loss_dollar": -25.58,
            "todays_gain_loss_percent": -3.48,
            "total_gain_loss_dollar": -111.41,
            "total_gain_loss_percent": -13.54,
            "percent_of_account": 1.12,
            "cost_basis_total": 822.97,
            "average_cost_basis": 411.49,
            "type": "Cash",
        },
    ]


def test_fidelity_positions_csv_parser_does_not_infer_account_number_as_index():
    text = (
        "Account number,Account name,Symbol,Description,Quantity,Last price,Last price change,"
        "Current value,Today's gain/loss dollar,Today's gain/loss percent,Total gain/loss dollar,"
        "Total gain/loss percent,Percent of account,Cost basis total,Average cost basis,Type\n"
        "X83851048,Individual - TOD,SPAXX**,HELD IN MONEY MARKET,,,,$335.70,,,,,0.53%,,,Cash,\n"
        "X83851048,Individual - TOD,GLD,SPDR GOLD TR GOLD SHS,2,$355.78,-$12.79,$711.56,"
        "-$25.58,-3.48%,-$111.41,-13.54%,1.12%,$822.97,$411.49,Cash,\n"
    )

    positions = parse_fidelity_positions_csv(text.encode("utf-8"))

    assert [position["symbol"] for position in positions] == ["SPAXX", "GLD"]
    assert positions[0]["type"] == "Cash"
    assert positions[0]["current_value"] == 335.7
    assert positions[1]["type"] == "Cash"
    assert positions[1]["total_gain_loss_percent"] == -13.54


def test_missing_columns_raise():
    with pytest.raises(ValueError, match=r"Not an A360 Positions file; missing columns: \['Qty', 'Cusip'\]"):
        parse_positions("Ticker,Market Value\nVTI,100\n", "A1", CONFIG)


def test_default_a360_file_loads(sources):
    positions = sources["positions"].get_positions("A2001")
    assert {p.ticker for p in positions} >= {"VTI", "SHV", "12345A678"}


def test_resolve_model_id_by_id_or_name(sources):
    s = sources["sales"]
    assert resolve_model_id(s, "Conservative 40/60") == "CON-40-60"
    assert resolve_model_id(s, "mod-60-40") == "MOD-60-40"
    with pytest.raises(KeyError):
        resolve_model_id(s, "nope")


def test_find_account_ids(sources):
    assert find_account_ids(sources, "smith") == ["A1001"]
    assert set(find_account_ids(sources, "all")) == {"A1001", "A1002", "A1003", "A2001"}
    assert find_account_ids(sources, "nobody") == []


def test_load_account_portfolio_weights_sum_to_one(sources):
    pf = load_account_portfolio(sources, "A1001")
    assert sum(pf.current_weights.values()) == pytest.approx(1.0)
    assert pf.model.model_id == "MOD-60-40"
    with pytest.raises(KeyError):
        load_account_portfolio(sources, "ZZZ")
