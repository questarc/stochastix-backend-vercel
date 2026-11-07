"""Plain-English methodology text for the "How it works" tab and the Advisor AI."""
from __future__ import annotations

METHODOLOGY_SECTIONS: list[dict[str, str]] = [
    {
        "title": "What it is",
        "body": (
            "A **Monte Carlo simulation** projects thousands of possible futures for a client's portfolio. "
            "Each future (a *trial*) draws a different sequence of yearly market returns. The headline result, "
            "**probability of success**, is the share of trials in which the money lasts until the plan end age."
        ),
    },
    {
        "title": "How it works",
        "body": (
            "1. Start from the account's current market value and asset-class weights.\n"
            "2. For every year until the plan end age, draw correlated returns for each asset class "
            "from the capital market assumptions (expected return, volatility, correlation).\n"
            "3. Before retirement, grow the portfolio and add the annual contribution.\n"
            "4. From retirement on, take the inflation-adjusted annual withdrawal, then apply the return.\n"
            "5. A portfolio that reaches zero stays at zero."
        ),
    },
    {
        "title": "How it calculates",
        "body": (
            "- **Sampling:** uniform random numbers come from a standard random generator (PRNG), "
            "Latin Hypercube (LHS) or a scrambled Sobol sequence.\n"
            "- **Correlation:** a Gaussian or Student-t copula (fat tails) links asset classes using the "
            "Cholesky factor of the correlation matrix.\n"
            "- **Distributions:** each asset class maps its uniforms to returns with a distribution "
            "(lognormal, normal, uniform, triangular, PERT, exponential or binomial) whose mean and "
            "standard deviation match the expected return and volatility.\n"
            "- **Reproducibility:** the same seed and settings always give the identical result."
        ),
    },
    {
        "title": "How to measure the results",
        "body": (
            "- **Probability of success:** share of trials with money left at the plan end age.\n"
            "- **Percentile bands:** P10, P25, P50 (median), P75 and P90 portfolio values by age.\n"
            "- **Median ending value (today's $):** the median final value divided by cumulative inflation.\n"
            "- **Depletion age:** for failed trials, the first age at which the portfolio reaches zero."
        ),
    },
    {
        "title": "Limitations",
        "body": (
            "Results are **projections, not guarantees** and not investment advice. They depend on the capital "
            "market assumptions, ignore taxes, fees and changes in spending, and assume yearly rebalancing to "
            "constant weights. Real markets can behave differently from any model."
        ),
    },
]


def find_section(topic: str) -> str:
    """Return the methodology text that best matches ``topic`` (all sections if no match)."""
    needle = (topic or "").strip().lower()
    matches = [s for s in METHODOLOGY_SECTIONS if needle and (needle in s["title"].lower() or needle in s["body"].lower())]
    chosen = matches or METHODOLOGY_SECTIONS
    return "\n\n".join(f"## {s['title']}\n{s['body']}" for s in chosen)
