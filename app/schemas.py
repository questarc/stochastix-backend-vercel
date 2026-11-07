"""Request / response models matching the shared API contract (snake_case)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_TRIALS = 50_000


class SourceInfo(BaseModel):
    name: str
    type: str
    supports_upload: bool


class Account(BaseModel):
    account_id: str
    client_name: str
    advisor: str
    age: int
    retirement_age: int
    plan_end_age: int
    annual_contribution: float
    annual_withdrawal: float
    risk_profile: str
    model_id: str


class Position(BaseModel):
    account_id: str
    ticker: str
    asset_class: str
    quantity: float
    price: float
    market_value: float


class ModelPortfolio(BaseModel):
    model_id: str
    model_name: str
    weights: dict[str, float]


class AccountPortfolio(BaseModel):
    source: str
    account: Account
    positions: list[Position]
    current_weights: dict[str, float]
    model: ModelPortfolio
    market_value: float


class FidelityPortfolioSummary(BaseModel):
    beginning_value: float | None = None
    ending_value: float | None = None
    reported_change: float | None = None
    total_including_other_holdings: float | None = None
    statement_period: str | None = None
    income_this_period: float | None = None
    income_year_to_date: float | None = None
    top_holdings_value: float | None = None
    top_holdings_percent: float | None = None


class FidelityAccountSummary(BaseModel):
    account_value: float | None = None
    change_in_account_value: float | None = None
    beginning_value: float | None = None
    additions: float | None = None
    subtractions: float | None = None
    transaction_costs: float | None = None
    investment_change: float | None = None
    ending_value: float | None = None
    accrued_interest: float | None = None
    income_this_period: float | None = None
    income_year_to_date: float | None = None
    top_holdings_value: float | None = None
    top_holdings_percent: float | None = None


class FidelityHolding(BaseModel):
    beginning_market_value: float | None = None
    quantity: float | None = None
    price_per_unit: float | None = None
    ending_market_value: float | None = None
    cost_basis: float | None = None
    unrealized_gain_loss: float | None = None
    estimated_annual_income: float | None = None
    estimated_yield_percent: float | None = None


class FidelityActivityTransaction(BaseModel):
    settlement_date: str
    security_name: str | None = None
    symbol_cusip: str | None = None
    description: str | None = None
    quantity: float | None = None
    price: float | None = None
    total_cost_basis: float | None = None
    transaction_cost: float | None = None
    amount: float | None = None


class FidelityCsvPosition(BaseModel):
    symbol: str
    description: str
    quantity: float | None = None
    last_price: float | None = None
    last_price_change: float | None = None
    current_value: float | None = None
    todays_gain_loss_dollar: float | None = None
    todays_gain_loss_percent: float | None = None
    total_gain_loss_dollar: float | None = None
    total_gain_loss_percent: float | None = None
    percent_of_account: float | None = None
    cost_basis_total: float | None = None
    average_cost_basis: float | None = None
    type: str | None = None


class EtradeStatement(BaseModel):
    filename: str
    page_count: int
    text: str
    character_count: int
    imported_at: str
    document_type: str
    transactions: list["EtradeTransaction"] = Field(default_factory=list)
    summary: "EtradeTransferSummary | None" = None
    portfolio_summary: "FidelityPortfolioSummary | None" = None
    account_summaries: list["FidelityAccountSummary"] = Field(default_factory=list)
    holdings: list["FidelityHolding"] = Field(default_factory=list)
    activity_transactions: list["FidelityActivityTransaction"] = Field(default_factory=list)


class EtradeTransaction(BaseModel):
    date: str
    type: str
    amount: float
    status: str


class EtradeTransferSummary(BaseModel):
    transaction_count: int
    total_amount: float
    completed_amount: float
    scheduled_amount: float


class SimulationRequest(BaseModel):
    account_id: str
    model_id: str | None = None
    retirement_age: int | None = None
    plan_end_age: int | None = None
    annual_contribution: float | None = None
    annual_withdrawal: float | None = None
    trials: int = Field(10_000, gt=0, le=MAX_TRIALS)
    seed: int | None = 42
    sampling: Literal["prng", "lhs", "sobol"] = "prng"
    copula: Literal["gaussian", "student_t"] = "gaussian"
    copula_df: float = 5
    distribution: str | None = None
    sample_paths: int = 0


class Bands(BaseModel):
    p10: list[float]
    p25: list[float]
    p50: list[float]
    p75: list[float]
    p90: list[float]


class Histogram(BaseModel):
    bin_edges: list[float]
    counts: list[int]


class DepletionHistogram(BaseModel):
    ages: list[int]
    counts: list[int]


class SimulationSettings(BaseModel):
    seed: int
    sampling: str
    copula: str
    copula_df: float
    trials: int
    years: int
    distributions: dict[str, str]


class SimulationResult(BaseModel):
    success_probability: float
    ages: list[int]
    bands: Bands
    ending_values_histogram: Histogram
    depletion_ages_histogram: DepletionHistogram
    median_ending_value_todays_dollars: float
    failed_trials: int
    median_depletion_age: float | None = None
    earliest_depletion_age: int | None = None
    settings: SimulationSettings
    sample_paths: list[list[float]] | None = None


class SimulationResponse(BaseModel):
    account_id: str
    model_id: str
    current: SimulationResult
    model: SimulationResult


class RebalanceRequest(BaseModel):
    account_id: str
    model_id: str | None = None
    drift_tolerance: float | None = Field(None, ge=0)
    min_trade_value: float | None = Field(None, ge=0)
    whole_shares: bool | None = None
    simulation: SimulationRequest | None = None


class Trade(BaseModel):
    ticker: str
    asset_class: str
    side: Literal["Sell", "Buy"]
    quantity: float
    price: float
    value: float


class AllocationRow(BaseModel):
    asset_class: str
    current: float
    model: float
    post_trade: float


class RebalanceResponse(BaseModel):
    account_id: str
    model_id: str
    model_name: str
    market_value: float
    max_drift: float
    rebalance_needed: bool
    trades: list[Trade]
    allocation: list[AllocationRow]
    total_sells: float
    total_buys: float
    uninvested_cash: float
    success_current: float | None = None
    success_after: float | None = None


class Order(BaseModel):
    order_id: str
    account_id: str
    model_id: str
    side: Literal["Sell", "Buy"]
    ticker: str
    asset_class: str
    quantity: float
    order_type: str = "Market"
    time_in_force: str = "Day"
    est_price: float
    est_value: float
    status: str = "Pending"
    created_at: str


class MethodologySection(BaseModel):
    title: str
    body: str


class LLMProvider(BaseModel):
    id: str
    label: str
    default_model: str
    models: list[str]
    api_key_env: str | None = None
    api_key_set: bool


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    provider: str
    model: str | None = None
    account_id: str | None = None
    model_id: str | None = None


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any]
    result: str


class AudioInfo(BaseModel):
    audio_id: str
    account_id: str | None = None
    client_name: str | None = None
    script: str
    voice: str
    duration_seconds: float
    created_at: str
    file_name: str
    url: str


class VoiceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    account_id: str | None = None


class ChatResponse(BaseModel):
    text: str
    tool_calls: list[ToolCall]
    orders_added: int
    orders: list[Order]
    audio: list[AudioInfo] = Field(default_factory=list)


class AgentSpec(BaseModel):
    system_prompt: str
    tools: list[dict[str, Any]]


class ToolRunRequest(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)


class ToolRunResponse(BaseModel):
    tool: str
    args: dict[str, Any]
    result: str
    orders_added: int
    orders: list[Order]
    audio: list[AudioInfo] = Field(default_factory=list)
