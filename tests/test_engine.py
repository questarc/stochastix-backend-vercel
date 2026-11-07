import numpy as np
import pytest
from scipy import stats

from app.engine import DISTRIBUTIONS, CapitalMarketAssumptions, SimulationInputs, make_marginal, simulate
from app.engine.sampling import correlated_uniforms, uniform_samples


@pytest.fixture
def cma():
    return CapitalMarketAssumptions.load("config/capital_market_assumptions.json")


def _inputs(**kw):
    base = dict(start_value=1_000_000, weights={"US Equity": 0.6, "US Bonds": 0.4}, current_age=55,
                retirement_age=65, plan_end_age=90, annual_contribution=10_000,
                annual_withdrawal=50_000, trials=2_000, seed=42)
    base.update(kw)
    return SimulationInputs(**base)


def test_same_seed_reproduces_identical_result(cma):
    a, b = simulate(_inputs(), cma), simulate(_inputs(), cma)
    assert a.success_probability == b.success_probability
    np.testing.assert_array_equal(a.paths, b.paths)


def test_different_seed_changes_paths(cma):
    a, b = simulate(_inputs(seed=1), cma), simulate(_inputs(seed=2), cma)
    assert not np.array_equal(a.paths, b.paths)


def test_none_seed_is_reported(cma):
    result = simulate(_inputs(seed=None), cma)
    assert isinstance(result.settings["seed"], int)


@pytest.mark.parametrize("sampling", ["lhs", "sobol"])
def test_qmc_sampling_is_reproducible(cma, sampling):
    a = simulate(_inputs(sampling=sampling, trials=1024), cma)
    b = simulate(_inputs(sampling=sampling, trials=1024), cma)
    assert a.success_probability == b.success_probability


def test_result_shapes_and_bands_are_ordered(cma):
    r = simulate(_inputs(), cma)
    assert r.paths.shape == (2_000, 36)
    assert len(r.ages) == 36 and r.ages[0] == 55 and r.ages[-1] == 90
    assert np.all(r.percentiles[10] <= r.percentiles[50]) and np.all(r.percentiles[50] <= r.percentiles[90])


def test_depletion_ages_only_for_failed_trials(cma):
    r = simulate(_inputs(annual_withdrawal=120_000), cma)
    failed = int(np.sum(r.ending_values <= 0))
    assert r.depletion_ages.size == failed
    assert np.all(r.depletion_ages >= 65)


def test_real_ending_value_deflated(cma):
    r = simulate(_inputs(), cma)
    np.testing.assert_allclose(r.ending_values_real, r.ending_values / (1 + cma.inflation) ** 35)


def test_no_withdrawal_always_succeeds(cma):
    r = simulate(_inputs(annual_withdrawal=0), cma)
    assert r.success_probability == 1.0


@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_distribution_mean_and_volatility(name):
    u = (np.arange(200_000) + 0.5) / 200_000
    x = make_marginal(name, 0.07, 0.16).ppf(u)
    assert x.mean() == pytest.approx(0.07, abs=2e-3)
    assert x.std() == pytest.approx(0.16, rel=2e-2)


def test_gaussian_copula_preserves_correlation():
    rng = np.random.default_rng(0)
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    u = correlated_uniforms(uniform_samples(rng, "prng", 20_000, 2), corr, 1, 2, "gaussian", 5)
    z = stats.norm.ppf(u[:, 0, :])
    assert np.corrcoef(z.T)[0, 1] == pytest.approx(0.8, abs=0.02)


def test_student_t_copula_has_fatter_joint_tails():
    rng = np.random.default_rng(0)
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    g = correlated_uniforms(uniform_samples(rng, "prng", 50_000, 2), corr, 1, 2, "gaussian", 4)[:, 0, :]
    rng = np.random.default_rng(0)
    t = correlated_uniforms(uniform_samples(rng, "prng", 50_000, 3), corr, 1, 2, "student_t", 4)[:, 0, :]
    joint = lambda u: np.mean((u[:, 0] < 0.01) & (u[:, 1] < 0.01))
    assert joint(t) > joint(g)


def test_with_distribution_overrides_all(cma):
    assert set(cma.with_distribution("normal").distributions.values()) == {"normal"}


@pytest.mark.parametrize(
    "kw, message",
    [
        (dict(sampling="bad"), "sampling"),
        (dict(copula="bad"), "copula"),
        (dict(copula="student_t", copula_df=2), "degrees of freedom"),
        (dict(annual_withdrawal=-1), "negative"),
        (dict(retirement_age=95), "Ages"),
        (dict(retirement_age=50), "Ages"),
        (dict(trials=0), "Trials"),
        (dict(weights={"Crypto": 1.0}), "Unknown asset classes"),
    ],
)
def test_validation_errors(cma, kw, message):
    with pytest.raises(ValueError, match=message):
        simulate(_inputs(**kw), cma)
