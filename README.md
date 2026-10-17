# eu-power-dispatch

**Optimisation models of European power markets: battery arbitrage (LP) and merit-order unit commitment with dual-based price formation (MIP).**

Author: Jamie North · Licence: MIT · Python 3.11+ · Pyomo + HiGHS (Gurobi optional)

---

## The question

> *How well does a cost-based merit-order model reproduce day-ahead power prices in Great Britain, and how do more wind or more batteries change price levels, volatility and the value of storage?*

The project builds two linked models on hourly ENTSO-E data:

1. **Battery arbitrage LP**: the maximum revenue a price-taking battery could earn from day-ahead price spreads, given power, energy, efficiency and cycling limits.
2. **Unit commitment MIP**: an aggregated thermal fleet dispatched against residual demand (load − wind − solar − nuclear), with on/off decisions, minimum stable generation and start-up costs. Hourly clearing prices come from the duals of the demand-balance constraint after fixing the commitment.

Modelled prices are validated against actual day-ahead prices, then the fleet and profiles are perturbed through scenario files (+10 GW wind, +5 GW / 10 GWh batteries).

---

## Model formulation

### Battery arbitrage (LP)

For hours $t = 1,\dots,T$ (one day, $\Delta t = 1\,\text{h}$), prices $\pi_t$, charge $c_t$, discharge $d_t$ and state of charge $s_t$:

$$
\max_{c,d,s} \; \sum_{t} \pi_t \,(d_t - c_t) \;-\; k \sum_t (c_t + d_t)
$$

subject to

$$
\begin{aligned}
0 \le c_t,\, d_t &\le P && \text{power limits} \\
\underline{S} \le s_t &\le \overline{S} && \text{SoC bounds} \\
s_t &= s_{t-1} + \eta\, c_t - \frac{d_t}{\eta} && \text{SoC dynamics},\ \eta = \sqrt{\eta_{\text{RT}}} \\
s_0 = s_T &= S_{\text{init}} && \text{cyclic day} \\
\sum_t d_t &\le N_{\text{cyc}} \, E && \text{cycle limit}
\end{aligned}
$$

$P$ is power (MW), $E$ energy capacity (MWh), $\eta_{\text{RT}}$ round-trip efficiency and $k \ge 0$ an optional throughput degradation cost. With $\eta < 1$ and non-negative prices, charging and discharging in the same hour is never optimal, so no binary is needed.

### Unit commitment and economic dispatch (MIP)

Units $u \in \mathcal{U}$ (committable subset $\mathcal{U}^C$), output $p_{u,t}$, on-status $x_{u,t}\in\{0,1\}$, start-up $y_{u,t}\in\{0,1\}$, curtailment $q_t \ge 0$, residual demand $D_t$:

$$
\min \; \sum_{u,t} MC_{u,t}\, p_{u,t} \;+\; \sum_{u \in \mathcal{U}^C,\, t} SC_u\, y_{u,t} \;+\; k^{\text{curt}} \sum_t q_t \;+\; k \sum_t (c_t + d_t)
$$

subject to

$$
\begin{aligned}
\sum_u p_{u,t} + (d_t - c_t) - q_t &= D_t \quad (\lambda_t) && \text{demand balance} \\
\underline{P}_u\, x_{u,t} \le p_{u,t} &\le \overline{P}_u\, x_{u,t} && u \in \mathcal{U}^C \\
0 \le p_{u,t} &\le \overline{P}_u && u \notin \mathcal{U}^C \\
y_{u,t} &\ge x_{u,t} - x_{u,t-1} && \text{start-up detection} \\
&\text{battery constraints as above} && \text{(optional storage block)}
\end{aligned}
$$

Marginal cost per unit and hour:

$$
MC_{u,t} = \underbrace{F_{f(u),t}\, HR_u}_{\text{fuel}} + \underbrace{C_t\, EF_u}_{\text{carbon}} + VOM_u
$$

with fuel price $F$ (currency/MWh$_\text{th}$), heat rate $HR$ (MWh$_\text{th}$/MWh$_\text{e}$), carbon price $C$ (currency/tCO$_2$) and emissions factor $EF$ (tCO$_2$/MWh$_\text{e}$). The fleet includes a price-taking import block and a value-of-lost-load slack generator so every hour is feasible.

### Price formation: MIP → fixed-commitment LP → duals

A MIP has no meaningful dual: integrality makes the feasible set non-convex. Prices are therefore recovered in two passes (O'Neill et al., 2005):

1. Solve the MIP to obtain optimal commitment $x^{\star}_{u,t}, y^{\star}_{u,t}$.
2. Fix $x = x^{\star}$, $y = y^{\star}$, relax integrality, and re-solve the remaining **LP**.
3. Take the clearing price as the dual of the balance constraint: $\lambda_t = \partial\, \text{cost} / \partial D_t$.

$\lambda_t$ is the cost of serving one more MWh given the committed fleet, which is the short-run marginal cost of the price-setting unit. Start-up costs are sunk once commitment is fixed and so do not appear in the price; in practice they are recovered through bid mark-ups or uplift. In oversupplied hours the price falls to $-k^{\text{curt}}$; when storage is the marginal resource, the price reflects its intertemporal opportunity cost.

See `clearing_prices_from_fixed_commitment` in [`merit_order_mip.py`](src/eu_power_dispatch/models/merit_order_mip.py).

---

## Data

| Source | Series | Notes |
|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) via `entsoe-py` | load, wind (on+offshore), solar, nuclear, day-ahead price | Hourly, converted to UTC. API key read from `ENTSOE_API_KEY`. |
| User-supplied CSV | daily gas and carbon prices | e.g. TTF / NBP and EUA / UKA. Forward-filled to hourly. |

### Raw data schema

Any source can be used as long as it produces this parquet at `data/raw/<ZONE>_<YYYYMMDD>_<YYYYMMDD>.parquet`:

| Column | Unit | Description |
|---|---|---|
| *index* | UTC | hourly `DatetimeIndex`, tz-aware, hour-beginning |
| `load_mw` | MW | actual total load |
| `wind_mw` | MW | wind onshore + offshore |
| `solar_mw` | MW | solar |
| `nuclear_mw` | MW | nuclear |
| `da_price` | currency/MWh | day-ahead price (`GBP` or `EUR`, set by `project.currency`) |

> **GB caveat.** GB data stopped flowing to ENTSO-E after Brexit (mid-2021). For later GB periods the loader raises a clear error; build the same schema from Elexon BMRS / NESO data instead. Any continental zone (`DE_LU`, `FR`, `NL`, ...) works directly by changing `project.zone` and the fleet file.

### Fuel CSV schema (`data/fuels/fuels_daily.csv`)

```
date,gas_price,carbon_price
2024-01-02,31.45,77.10
```

`gas_price` in currency/MWh$_\text{th}$, `carbon_price` in currency/tCO$_2$. Use `fuels.gas_fx` / `fuels.carbon_fx` in `base.yaml` to convert currencies, and `fuels.carbon_price_adder` for the GB Carbon Price Support.

No data is committed to the repository.

---

## How to run

```bash
git clone https://github.com/jlnorth/eu-power-dispatch.git
cd eu-power-dispatch
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your ENTSO-E key, then: export $(cat .env | xargs)
```

Pipeline (each step is a console script and also runnable as `python -m eu_power_dispatch.<module>`):

```bash
epd-fetch                                   # ENTSO-E -> data/raw/*.parquet
epd-prepare                                 # clean, UTC, residual demand, fuels -> data/processed/
epd-market                                  # day-by-day UC/ED -> outputs/base/modelled_prices.parquet
epd-validate --overlay-start 2024-01-15     # metrics + PNGs in outputs/base/
epd-battery --prices actual                 # battery arbitrage on actual prices
epd-battery --prices modelled               # ... or on modelled prices
epd-scenario config/scenarios/high_wind.yaml
epd-scenario config/scenarios/more_storage.yaml
```

Useful flags: `--config path/to/base.yaml`, `--days N` (solve only the first N days), `--run-name NAME`.

### Switching solver

```yaml
# config/base.yaml
solver:
  name: gurobi        # highs (default) | gurobi | any Pyomo solver plugin name
  mip_gap: 0.001
  time_limit_s: 120
```

`pip install -e ".[gurobi]"` and a valid licence are needed for Gurobi.

### Tests

```bash
pytest
```

Tests use small synthetic data and run offline. They check SoC bounds, no simultaneous charge/discharge, positive revenue on a low-then-high price series, energy balance and the cycle limit for the battery, and, for the market, hourly demand balance, merit-order dispatch, price equal to the marginal unit's cost, VoLL pricing under scarcity, minimum stable generation and curtailment pricing.

---

## Project structure

```
eu-power-dispatch/
├── config/
│   ├── base.yaml                 # dates, zone, solver, paths, standalone battery
│   ├── fleet_gb.yaml             # aggregated thermal units, imports, VoLL, market storage
│   └── scenarios/
│       ├── high_wind.yaml        # +10 GW wind
│       └── more_storage.yaml     # +5 GW / 10 GWh batteries
├── src/eu_power_dispatch/
│   ├── config.py                 # YAML -> validated dataclasses
│   ├── solver.py                 # solver factory (HiGHS / Gurobi) and optimality check
│   ├── data/
│   │   ├── entsoe_loader.py      # ENTSO-E download -> raw parquet
│   │   ├── fuels.py              # daily gas + carbon -> hourly
│   │   └── prepare.py            # UTC alignment, gap filling, residual demand
│   ├── models/
│   │   ├── battery_lp.py         # battery arbitrage LP + reusable battery block
│   │   └── merit_order_mip.py    # UC/ED MIP and fixed-commitment dual pricing
│   ├── run_battery.py            # CLI: battery day by day
│   ├── run_market.py             # CLI: market day by day
│   ├── validate.py               # metrics and plots
│   └── scenarios.py              # scenario overrides and comparison
├── tests/
└── outputs/                      # generated, gitignored
```

---

## Results

> **Placeholder.** Results for GB 2024 will be added once the full-year run is complete.

| Metric | Value |
|---|---|
| MAE (currency/MWh) | _TBD_ |
| RMSE (currency/MWh) | _TBD_ |
| Correlation | _TBD_ |
| Mean bias (currency/MWh) | _TBD_ |

<!-- RESULTS: price-duration curve -->
`![Price-duration curve](docs/img/price_duration_curve.png)` *(placeholder)*

<!-- RESULTS: two-week overlay -->
`![Two-week overlay](docs/img/price_overlay.png)` *(placeholder)*

<!-- RESULTS: scatter -->
`![Modelled vs actual scatter](docs/img/price_scatter.png)` *(placeholder)*

<!-- RESULTS: scenarios -->
| Scenario | Δ mean price | Δ price std | Δ battery revenue |
|---|---|---|---|
| +10 GW wind | _TBD_ | _TBD_ | _TBD_ |
| +5 GW / 10 GWh storage | _TBD_ | _TBD_ | _TBD_ |

---

## Assumptions and limitations

- **Perfect foresight.** Each day is optimised with full knowledge of that day's demand, renewables and prices. Battery revenues are an upper bound on what a forecast-driven strategy would capture.
- **Day-by-day horizon.** Days are solved independently; storage returns to its initial SoC at midnight and commitment state is carried over, but there is no look-ahead across days.
- **Aggregated fleet.** Units are capacity bands, not individual plants. A band's binary on/off plus minimum stable generation is a coarse proxy for many units starting independently. No ramp rates, minimum up/down times or part-load efficiency curves.
- **No scarcity pricing or mark-ups.** Prices are pure short-run marginal cost. Real prices include scarcity rents, start-up cost recovery, strategic bidding and balancing risk premia, so the model is expected to under-predict peaks.
- **Single zone, copper plate.** No internal transmission constraints. Imports are a single price-taking block with a fixed offer price, not a coupled neighbouring market.
- **Exogenous must-run.** Wind, solar and nuclear are taken as historic actuals; curtailment is the only flexibility on that side. Hydro and pumped storage are not modelled separately.
- **Static fuel costs** for biomass; daily gas and carbon prices applied flat within each day.
- **Scenario scaling** of wind uses the historic hourly capacity-factor shape, which ignores the different profile of new (e.g. offshore) capacity.

---

## Roadmap

- [ ] **Cross-border coupling**: multi-zone model with interconnector NTC limits and flow-based pricing (GB, FR, NL, BE, NO, DK, IE), replacing the import price-taker block.
- [ ] **Forecast-driven battery dispatch**: rolling-horizon optimisation on price forecasts (naive, gradient-boosted, quantile) to measure the gap to perfect foresight.
- [ ] Minimum up/down times, ramp limits and multi-segment heat-rate curves.
- [ ] Rolling multi-day horizon with overlap to remove midnight boundary effects.
- [ ] Scarcity adder / uplift calibration to close the peak-price gap.
- [ ] Ancillary services co-optimisation for storage (frequency response, reserve).
- [ ] CI workflow running tests and a small end-to-end sample.

---

## Licence

MIT, see [LICENSE](LICENSE).
