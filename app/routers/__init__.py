from . import (agent_tools, chat, etrade, fidelity, llm, market, methodology, orders, portfolio, rebalance, simulations, sources,
               time_series, voice)

ALL_ROUTERS = [
    sources.router,
    portfolio.router,
    simulations.router,
    rebalance.router,
    orders.router,
    methodology.router,
    llm.router,
    chat.router,
    agent_tools.router,
    voice.router,
    market.router,
    etrade.router,
    fidelity.router,
    time_series.router,
]
