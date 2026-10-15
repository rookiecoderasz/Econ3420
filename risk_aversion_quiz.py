"""One portfolio-based certainty-equivalent choice for Assignment One, Q5.

Show the scenario and menu: python risk_aversion_quiz.py --show-scenarios
Enter your own A-E choice: python risk_aversion_quiz.py
Import your recorded choice: python risk_aversion_quiz.py --responses data/ce_responses.csv

The response CSV has trial_id,choice and exactly one row: trial_id 1 and the
student's selected letter A-E. The saved Q4 portfolio and dated safe rate are
read from outputs/q5_portfolio.json, or a supplied --portfolio path.
Choose the offered guaranteed final wealth closest to your certainty
equivalent for that portfolio. The implied gamma is a coarse classroom
estimate from a limited menu, not an unrestricted measure of preferences.
AI must not choose your answer; missing answers are never supplied.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

STARTING_WEALTH = 10_000
SCHEMA_VERSION = 2
CHOICES = tuple("ABCDE")
METADATA_FIELDS = (
    "currency", "safe_rate_date", "safe_rate_date_kind", "first_return_month",
    "last_return_month", "data_fingerprint",
)
PORTFOLIO_NUMBERS = ("annual_premium", "annual_volatility", "annual_safe_rate")
# z-score of the lowest-10% boundary shown with the menu.
LOWEST_DECILE_Z = float(norm.ppf(0.9))


def finite_number(value, name):
    """Read a finite real number, rejecting booleans and missing values."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def normalize_portfolio_inputs(payload):
    """Validate and normalize the schema-2 Q4 portfolio snapshot.

    Required fields: schema_version, five labelled weights, annual_premium,
    annual_volatility, annual_safe_rate, currency, safe_rate_date,
    safe_rate_date_kind, first_return_month, last_return_month, n_months and
    data_fingerprint (SHA256 of the matching monthly excess-return CSV).
    Returns and volatility use decimal annual units. The annual mean is
    annual_safe_rate + annual_premium. A nonpositive premium is valid data,
    but cannot be used to construct an interior-allocation Q5 menu.
    """
    if not isinstance(payload, dict):
        raise ValueError("The portfolio inputs must be a JSON object.")
    required = {"schema_version", "weights", "n_months", *PORTFOLIO_NUMBERS,
                *METADATA_FIELDS}
    missing = required.difference(payload)
    if missing:
        raise ValueError("The portfolio inputs are missing: " + ", ".join(sorted(missing)) + ".")
    version = finite_number(payload["schema_version"], "schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError("The portfolio inputs must use schema_version 2.")
    weights = payload["weights"]
    if not isinstance(weights, dict) or len(weights) != 5:
        raise ValueError("Supply weights for exactly five labelled stocks.")
    normalized_weights = {}
    for label, value in weights.items():
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            raise ValueError("Every stock weight needs a nonblank, unpadded text label.")
        weight = finite_number(value, f"weight for {label}")
        if weight < 0 or weight > 1 + 1e-8:
            raise ValueError("Each stock weight must be nonnegative and no greater than 1.")
        normalized_weights[label] = weight
    if not math.isclose(math.fsum(normalized_weights.values()), 1, rel_tol=0, abs_tol=1e-8):
        raise ValueError("The five stock weights must total 1.")
    result = {"schema_version": SCHEMA_VERSION, "weights": normalized_weights}
    for field in PORTFOLIO_NUMBERS:
        result[field] = finite_number(payload[field], field)
    if result["annual_volatility"] <= 0:
        raise ValueError("annual_volatility must be positive.")
    variance = result["annual_volatility"] * result["annual_volatility"]
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("The annual variance must be positive and finite.")
    if result["annual_safe_rate"] <= -1:
        raise ValueError("annual_safe_rate must be greater than -1.")
    if not math.isfinite(result["annual_safe_rate"] + result["annual_premium"]):
        raise ValueError("The annual portfolio mean must be finite.")
    months = finite_number(payload["n_months"], "n_months")
    if months < 60 or months != int(months):
        raise ValueError("n_months must be an integer of at least 60.")
    result["n_months"] = int(months)
    for field in METADATA_FIELDS:
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be nonblank text.")
        result[field] = value
    if not re.fullmatch(r"[0-9a-fA-F]{64}", result["data_fingerprint"]):
        raise ValueError("data_fingerprint must be a 64-character SHA256 hex digest.")
    result["data_fingerprint"] = result["data_fingerprint"].lower()
    return result


def load_portfolio_inputs(path):
    """Load the saved schema-2 Q4 portfolio and dated safe-rate snapshot."""
    with Path(path).open(encoding="utf-8") as source:
        return normalize_portfolio_inputs(json.load(source))


def build_menu(portfolio_inputs):
    """Construct five CE options from this portfolio's premium and variance.

    Gamma equals multiplier * G, where G = premium / variance. The default
    multipliers 1.25, 2, 3, 4, 5 imply risky shares 80%, 50%, 33.33%, 25%, 20%.
    No option may offer sure wealth below the portfolio's lowest-10% outcome
    (or a nonpositive amount). For a high-Sharpe portfolio this caps the
    largest multiplier, and all five are then spaced evenly above one.
    Amounts stay unrounded.
    """
    inputs = normalize_portfolio_inputs(portfolio_inputs)
    premium = inputs["annual_premium"]
    safe_rate = inputs["annual_safe_rate"]
    variance = inputs["annual_volatility"] ** 2
    if premium <= 0:
        raise ValueError("Q4's annual expected excess return must be positive for this "
                         "menu. With a nonpositive premium, positive risk aversion "
                         "cannot give a risky share strictly between 0 and 1; "
                         "explain the safe-asset outcome instead.")
    g = premium / variance
    # c_m >= mu_P - z * sigma_P  <=>  m <= 2 z sigma_P / premium. Without this
    # cap, a high-Sharpe portfolio offers sure losses that are worse than the
    # gamble's lowest-10% outcome, which almost nobody would accept.
    floor_cap = 2.0 * LOWEST_DECILE_Z * inputs["annual_volatility"] / premium
    if floor_cap <= 1.05:
        floor_cap = 1.25  # Sharpe above about 2.4: the floor needs a share above 100%.
    max_multiplier = min(5.0, floor_cap, 2.0 + (1.0 + safe_rate) / premium)
    if max_multiplier >= 5.0:
        multipliers = np.array([1.25, 2.0, 3.0, 4.0, 5.0])
    else:
        min_multiplier = min(1.25, 1.0 + (max_multiplier - 1.0) / 5.0)
        multipliers = np.linspace(min_multiplier, max_multiplier, 5)
    # This form avoids subtracting two nearly equal portfolio returns and
    # avoids introducing precision loss by rounding sure wealth to cents.
    utility_premiums = premium * (1.0 - multipliers / 2.0)
    certainty_equivalents = safe_rate + utility_premiums
    wealth = STARTING_WEALTH * ((1.0 + safe_rate) + utility_premiums)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        gammas = multipliers * g
    menu = pd.DataFrame({
        "choice": CHOICES,
        "multiplier": multipliers,
        "gamma": gammas,
        "certainty_equivalent": certainty_equivalents,
        "certainty_equivalent_wealth": wealth,
    })
    numeric = menu.drop(columns="choice").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (menu["gamma"] <= g).any():
        raise ValueError("The supplied portfolio is outside finite numerical range for the menu.")
    if (menu["certainty_equivalent_wealth"] <= 0).any():
        raise ValueError("The supplied portfolio cannot produce positive sure wealth at numerical precision.")
    if not (np.diff(menu["certainty_equivalent_wealth"].to_numpy()) < 0).all():
        raise ValueError("The premium is too small to give five distinguishable sure-wealth "
                         "options at numerical precision. Do not invent a student choice.")
    return menu


def normalize_choice(value):
    """Validate the student's explicit letter; never infer a missing answer."""
    choice = str(value).strip().upper()
    if choice not in CHOICES:
        raise ValueError("The student choice must be one letter A-E; blanks are not accepted.")
    return choice


def close_number(actual, expected):
    """Allow CSV/JSON floating-point round trips while keeping checks tight."""
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)


def matching_portfolio(saved, current):
    """Reject a changed portfolio, safe rate, or underlying return dataset."""
    saved = normalize_portfolio_inputs(saved)
    current = normalize_portfolio_inputs(current)
    if saved["weights"].keys() != current["weights"].keys():
        return False
    numbers_match = all(close_number(saved[field], current[field])
                        for field in PORTFOLIO_NUMBERS)
    weights_match = all(close_number(saved["weights"][stock], current["weights"][stock])
                        for stock in saved["weights"])
    metadata_match = (saved["n_months"] == current["n_months"]
                      and all(saved[field] == current[field] for field in METADATA_FIELDS))
    return numbers_match and weights_match and metadata_match


def summarize_trials(trials):
    """Validate the one recorded portfolio choice and recompute its gamma.

    The row must contain the schema-2 portfolio snapshot and explicit choice,
    moments, multiplier, CE return and CE wealth. The choice is never filled
    or inferred. Existing gamma is not trusted: it is recomputed from the
    validated multiplier and premium/variance, which is algebraically the CE
    formula but avoids cancellation for very small premiums.
    Return {'estimates': DataFrame, 'gamma': float}; there is no median.
    """
    numeric = ["schema_version", "trial_id", "annual_premium", "annual_safe_rate",
               "mean_return", "volatility", "certainty_equivalent",
               "certainty_equivalent_wealth", "multiplier", "gamma"]
    required = {*numeric, "choice", "portfolio_inputs_json"}
    if not isinstance(trials, pd.DataFrame) or not required.issubset(trials):
        raise ValueError("Supply the complete schema-2 portfolio-choice row, "
                         "including its explicit choice and portfolio_inputs_json.")
    if len(trials) != 1:
        raise ValueError("Supply exactly one portfolio choice, numbered trial_id 1.")
    result = trials.copy().reset_index(drop=True)
    for field in numeric:
        result[field] = result[field].map(lambda value: finite_number(value, field)).astype(float)
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("All numeric choice inputs must be finite.")
    row = result.iloc[0]
    if row["schema_version"] != SCHEMA_VERSION or row["trial_id"] != 1:
        raise ValueError("The saved answer must use schema_version 2 and trial_id 1.")
    result["schema_version"] = result["schema_version"].astype(int)
    result["trial_id"] = result["trial_id"].astype(int)
    try:
        inputs = normalize_portfolio_inputs(json.loads(row["portfolio_inputs_json"]))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("The saved portfolio_inputs_json must contain the portfolio snapshot.") from error
    choice = normalize_choice(row["choice"])
    expected = build_menu(inputs).set_index("choice").loc[choice]
    fields = {
        "annual_premium": inputs["annual_premium"],
        "annual_safe_rate": inputs["annual_safe_rate"],
        "mean_return": inputs["annual_safe_rate"] + inputs["annual_premium"],
        "volatility": inputs["annual_volatility"],
        "multiplier": float(expected["multiplier"]),
        "certainty_equivalent": float(expected["certainty_equivalent"]),
        "certainty_equivalent_wealth": float(expected["certainty_equivalent_wealth"]),
    }
    for field, value in fields.items():
        if not close_number(float(row[field]), value):
            raise ValueError(f"The saved {field} does not match the selected portfolio option.")
        result.loc[0, field] = value
    for field in ["recorded_at", "entry_method"]:
        if field in result and (not isinstance(row[field], str) or not row[field].strip()):
            raise ValueError(f"Saved {field} provenance must be nonblank text.")
    variance = inputs["annual_volatility"] ** 2
    if "variance" in result:
        saved_variance = finite_number(row["variance"], "variance")
        if not close_number(saved_variance, variance):
            raise ValueError("The saved variance does not match the portfolio snapshot.")
    gamma = float(expected["multiplier"]) * (inputs["annual_premium"] / variance)
    if not math.isfinite(gamma) or gamma <= inputs["annual_premium"] / variance:
        raise ValueError("The implied gamma must exceed premium/variance.")
    result.loc[0, "choice"] = choice
    result.loc[0, "gamma"] = gamma
    result.loc[0, "variance"] = variance
    result.loc[0, "portfolio_inputs_json"] = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return {"estimates": result, "gamma": gamma}


def load_preferences(path, portfolio_inputs=None):
    """Validate the saved single choice and optionally reject stale Q4 inputs.

    Legacy freeform or five-case answers are rejected; no old answer is
    mapped to a new option. Pass the current portfolio snapshot when using
    gamma in Q5-Q6 so changed data, stocks or cash rates fail.
    """
    result = summarize_trials(pd.read_csv(path))["estimates"]
    if portfolio_inputs is not None:
        saved = json.loads(result.loc[0, "portfolio_inputs_json"])
        if not matching_portfolio(saved, portfolio_inputs):
            raise ValueError("The saved Q5 answer belongs to different Q4 portfolio "
                             "inputs or a different safe rate. Record a new student choice.")
    return result


def record_choice(portfolio_inputs, choice, entry_method):
    """Derive estimates from an explicit student choice and preserve its inputs."""
    inputs = normalize_portfolio_inputs(portfolio_inputs)
    choice = normalize_choice(choice)
    option = build_menu(inputs).set_index("choice").loc[choice]
    record = {
        "schema_version": SCHEMA_VERSION,
        "trial_id": 1,
        "choice": choice,
        "annual_premium": inputs["annual_premium"],
        "annual_safe_rate": inputs["annual_safe_rate"],
        "mean_return": inputs["annual_safe_rate"] + inputs["annual_premium"],
        "volatility": inputs["annual_volatility"],
        "certainty_equivalent": float(option["certainty_equivalent"]),
        "certainty_equivalent_wealth": float(option["certainty_equivalent_wealth"]),
        "multiplier": float(option["multiplier"]),
        "gamma": float(option["gamma"]),
        "portfolio_inputs_json": json.dumps(inputs, sort_keys=True, separators=(",", ":")),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "entry_method": entry_method,
    }
    return pd.DataFrame([record])


def present_scenario(portfolio_inputs):
    """Show this portfolio's normal-return bands and five sure-wealth options."""
    inputs = normalize_portfolio_inputs(portfolio_inputs)
    menu = build_menu(inputs)
    mean = inputs["annual_safe_rate"] + inputs["annual_premium"]
    volatility = inputs["annual_volatility"]
    low, high = norm.ppf([0.1, 0.9], loc=mean, scale=volatility)
    currency = inputs["currency"]

    def wealth(ret):
        return f"{STARTING_WEALTH * (1 + ret):,.2f}"

    rows = [
        ("Lowest 10%", f"below {low:+.1%}", f"below {wealth(low)}"),
        ("Middle 80%", f"{low:+.1%} to {high:+.1%}", f"{wealth(low)} to {wealth(high)}"),
        ("Highest 10%", f"above {high:+.1%}", f"above {wealth(high)}"),
    ]
    headers = ("Share of outcomes", "Annual return", f"Final wealth ({currency})")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(3)]

    def line(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    rule = "  ".join("-" * width for width in widths)
    print(f"\nYour Q4 portfolio: one year, starting wealth {STARTING_WEALTH:,.2f} {currency}")
    print(f"Mean return {mean:.2%}; standard deviation {volatility:.2%}. "
          f"Safe rate {inputs['annual_safe_rate']:.2%}, dated {inputs['safe_rate_date']}.")
    print("The return ranges use a normal-distribution classroom approximation.")
    print(rule)
    print(line(headers))
    print(rule)
    for row in rows:
        print(line(row))
    print(rule)
    print("Choose the guaranteed final wealth closest to your certainty equivalent:")
    # A small positive premium can make otherwise distinct choices look identical
    # at cent precision. Increase display precision rather than show equal options.
    for digits in range(2, 16):
        labels = [f"{value:,.{digits}f}" for value in menu["certainty_equivalent_wealth"]]
        if len(set(labels)) == len(CHOICES):
            break
    else:
        raise ValueError("The sure-wealth options cannot be distinguished at display precision.")
    for choice, amount in zip(menu["choice"], labels):
        print(f"  {choice}. {amount} {currency}")
    if digits == 2:
        print("Amounts are rounded to cents for display; calculations retain unrounded values.")
    else:
        print(f"Amounts use {digits} decimal places to distinguish this small-premium menu; "
              "calculations retain unrounded values.")
    print("This limited menu gives a coarse classroom estimate of your preferences.\n")


def collect_choices(portfolio_inputs):
    """Ask for the student's single A-E choice and print its gamma immediately."""
    present_scenario(portfolio_inputs)
    while True:
        try:
            choice = normalize_choice(input("Your choice (A-E): "))
        except ValueError as error:
            print(f"{error} No student answer is suggested.")
            continue
        record = record_choice(portfolio_inputs, choice, "student multiple-choice keyboard entry")
        gamma = summarize_trials(record)["gamma"]
        print(f"Implied risk aversion (gamma): {gamma:.8g}")
        return record


def read_responses(path, portfolio_inputs):
    """Import exactly one recorded student A-E answer, without guessing."""
    responses = pd.read_csv(path)
    if not {"trial_id", "choice"}.issubset(responses):
        raise ValueError("The response CSV needs trial_id and choice columns (A-E).")
    if len(responses) != 1:
        raise ValueError("The response CSV must contain exactly one student choice.")
    trial_id = finite_number(responses.iloc[0]["trial_id"], "trial_id")
    if trial_id != 1:
        raise ValueError("The response CSV must use trial_id 1.")
    choice = normalize_choice(responses.iloc[0]["choice"])
    return record_choice(portfolio_inputs, choice,
                         "imported student-choice CSV; timestamp is import time")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", type=Path, default=Path("outputs/q5_portfolio.json"),
                        help="Saved Q4 portfolio and dated safe-rate inputs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--show-scenarios", "--show-scenario", action="store_true",
                      help="Show the portfolio distribution and menu without collecting an answer.")
    mode.add_argument("--responses", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/certainty_equivalents.csv"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace an existing results file.")
    args = parser.parse_args(argv)
    try:
        if not args.show_scenarios and args.output.exists() and not args.overwrite:
            raise ValueError("The output already exists. Choose another --output path "
                             "or deliberately use --overwrite.")
        if not args.portfolio.is_file():
            raise ValueError(f"{args.portfolio} was not found. Complete and render Questions 1-4 "
                             "first; the Question 4 cell saves this file.")
        inputs = load_portfolio_inputs(args.portfolio)
        build_menu(inputs)  # Validate the positive-premium menu before asking for an answer.
        if args.show_scenarios:
            present_scenario(inputs)
            return 0
        print("A certainty equivalent is the sure final wealth you value equally to "
              "your risky portfolio. Select the closest offered amount. "
              "AI must not invent or choose your answer.")
        trials = (read_responses(args.responses, inputs) if args.responses
                  else collect_choices(inputs))
        summary = summarize_trials(trials)
        estimates = summary["estimates"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against a file created after the check above.
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8", newline="") as destination:
            estimates.to_csv(destination, index=False)
        row = estimates.iloc[0]
        print(f"\nChoice {row['choice']}: sure final wealth "
              f"{row['certainty_equivalent_wealth']:,.2f} {inputs['currency']}.")
        print(f"Gamma: {summary['gamma']:.8g}. Saved inputs and estimate to {args.output}.")
        print("Use this gamma with the same Q4 portfolio inputs in Question 6.")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nStopped before completing the exercise.")
        return 1
    except (ValueError, OSError, OverflowError) as error:
        print(f"Cannot complete the quiz: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
