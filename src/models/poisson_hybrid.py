"""
src/models/poisson_hybrid.py
----------------------------
Dixon-Coles Poisson model with calibrated Elo ratings (Hicruben approach).

References:
  - Dixon & Coles (1997) — bivariate Poisson with low-score correction
  - Hicruben/world-cup-2026-prediction-model — calibrated Elo + DC model

Key formula:
  expected_goals = max(0.3, min(3.5, 1.35 + (elo_diff + home_bonus) / 400))
"""

import math
import numpy as np

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DC_RHO = -0.13          # Dixon-Coles correction parameter
MAX_GOALS = 10           # Max goals to consider for probability sum
HOST_NATIONS = {"Mexico", "USA", "Canada"}
HOME_ADV = 75            # Elo home advantage for host nations

# Calibrated Elo ratings (from 913 international matches, Hicruben)
ELO_RATINGS = {
    "Spain": 2010, "France": 2009, "England": 1993, "Argentina": 1976,
    "Brazil": 1955, "Portugal": 1945, "Germany": 1926, "Italy": 1901,
    "Netherlands": 1894, "Norway": 1880, "Belgium": 1878, "Colombia": 1878,
    "Morocco": 1874, "Croatia": 1852, "Senegal": 1848, "Mexico": 1834,
    "Uruguay": 1831, "Ecuador": 1829, "USA": 1826, "Japan": 1825,
    "Denmark": 1795, "Australia": 1772, "South Korea": 1760, "Sweden": 1752,
    "Iran": 1747, "Canada": 1740, "Turkey": 1731, "Poland": 1731,
    "Ivory Coast": 1732, "Austria": 1718, "Serbia": 1714, "Algeria": 1704,
    "Egypt": 1695, "Paraguay": 1681, "Tunisia": 1680, "Nigeria": 1671,
    "Wales": 1688, "Scotland": 1663, "Ghana": 1659, "Saudi Arabia": 1657,
    "Czech Republic": 1651, "DR Congo": 1650, "Cameroon": 1633,
    "Uzbekistan": 1633, "Venezuela": 1625, "Panama": 1615,
    "Bosnia & Herzegovina": 1602, "Iraq": 1599, "Cape Verde": 1599,
    "South Africa": 1591, "New Zealand": 1591, "Qatar": 1592,
    "Curacao": 1548, "Jordan": 1548, "Haiti": 1537, "Jamaica": 1514,
    "Honduras": 1497, "El Salvador": 1438, "Trinidad & Tobago": 1429,
    "Guatemala": 1416,
}

# Team groups for 2026 World Cup
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia & Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# 24 completed matches (as of 2026-06-18)
COMPLETED_MATCHES = [
    {"home": "Mexico", "away": "South Africa", "home_score": 2, "away_score": 0},
    {"home": "South Korea", "away": "Czech Republic", "home_score": 2, "away_score": 1},
    {"home": "Canada", "away": "Bosnia & Herzegovina", "home_score": 1, "away_score": 1},
    {"home": "Qatar", "away": "Switzerland", "home_score": 1, "away_score": 1},
    {"home": "Brazil", "away": "Morocco", "home_score": 1, "away_score": 1},
    {"home": "USA", "away": "Paraguay", "home_score": 4, "away_score": 1},
    {"home": "Haiti", "away": "Scotland", "home_score": 0, "away_score": 1},
    {"home": "Australia", "away": "Turkey", "home_score": 2, "away_score": 0},
    {"home": "Germany", "away": "Curacao", "home_score": 7, "away_score": 1},
    {"home": "Netherlands", "away": "Japan", "home_score": 2, "away_score": 2},
    {"home": "Ivory Coast", "away": "Ecuador", "home_score": 1, "away_score": 0},
    {"home": "Sweden", "away": "Tunisia", "home_score": 5, "away_score": 1},
    {"home": "Spain", "away": "Cape Verde", "home_score": 0, "away_score": 0},
    {"home": "Belgium", "away": "Egypt", "home_score": 1, "away_score": 1},
    {"home": "Saudi Arabia", "away": "Uruguay", "home_score": 1, "away_score": 1},
    {"home": "Iran", "away": "New Zealand", "home_score": 2, "away_score": 2},
    {"home": "France", "away": "Senegal", "home_score": 3, "away_score": 1},
    {"home": "Iraq", "away": "Norway", "home_score": 1, "away_score": 4},
    {"home": "Argentina", "away": "Algeria", "home_score": 3, "away_score": 0},
    {"home": "Austria", "away": "Jordan", "home_score": 3, "away_score": 1},
    {"home": "Portugal", "away": "DR Congo", "home_score": 1, "away_score": 1},
    {"home": "England", "away": "Croatia", "home_score": 4, "away_score": 2},
    {"home": "Ghana", "away": "Panama", "home_score": 1, "away_score": 0},
    {"home": "Uzbekistan", "away": "Colombia", "home_score": 1, "away_score": 3},
]


# ─────────────────────────────────────────────────────────────
# 1. DIXON-COLES POISSON MODEL
# ─────────────────────────────────────────────────────────────

def expected_goals(rating, opponent, home_bonus=0):
    """
    Expected goals from Elo rating difference (Hicruben formula).
    lambda = 1.35 + diff / 400, clamped to [0.3, 3.5]
    """
    diff = (rating + home_bonus) - opponent
    lam = 1.35 + diff / 400
    return max(0.3, min(3.5, lam))


def dc_tau(h, a, lam, mu, rho=DC_RHO):
    """Dixon-Coles correction for low-score outcomes."""
    if h == 0 and a == 0:
        return 1 - lam * mu * rho
    if h == 0 and a == 1:
        return 1 + lam * rho
    if h == 1 and a == 0:
        return 1 + mu * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1


def poisson_pmf(k, lam):
    """Poisson probability mass function."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def dixon_coles_match_prob(home, away, elo_dict=ELO_RATINGS):
    """
    Dixon-Coles bivariate Poisson prediction.
    Returns dict with W/D/L probabilities and expected goals.
    """
    eH = elo_dict.get(home, 1600)
    eA = elo_dict.get(away, 1600)

    # Home bonus only for host nations
    hb = (HOME_ADV if home in HOST_NATIONS else 0) - (HOME_ADV if away in HOST_NATIONS else 0)

    home_xg = expected_goals(eH, eA, hb)
    away_xg = expected_goals(eA, eH, -hb / 2)

    # Bivariate Poisson with Dixon-Coles correction
    ph = 0.0
    pd = 0.0
    pa = 0.0
    best_score = "1-0"
    best_prob = 0.0

    for h in range(MAX_GOALS + 1):
        pH = poisson_pmf(h, home_xg)
        for a in range(MAX_GOALS + 1):
            tau = dc_tau(h, a, home_xg, away_xg)
            p = pH * poisson_pmf(a, away_xg) * tau
            score_str = f"{h}-{a}"

            if h > a:
                ph += p
            elif h == a:
                pd += p
            else:
                pa += p

            if p > best_prob:
                best_prob = p
                best_score = score_str

    total = ph + pd + pa
    return {
        "ph": round(ph / total, 6),
        "pd": round(pd / total, 6),
        "pa": round(pa / total, 6),
        "home_xg": round(home_xg, 3),
        "away_xg": round(away_xg, 3),
        "most_likely_score": best_score,
        "most_likely_score_prob": round(best_prob, 4),
    }


# ─────────────────────────────────────────────────────────────
# 2. EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_predictions(predictions, actual_results):
    """
    Evaluate predictions against actual results.
    Calculates accuracy, log-loss, Brier score, RPS, and draw accuracy.
    """
    if len(predictions) != len(actual_results):
        raise ValueError("Predictions and results must have same length")

    n = len(predictions)
    label_map = {"W": 0, "D": 1, "L": 2}
    inv_map = {0: "W", 1: "D", 2: "L"}

    correct = 0
    log_loss_sum = 0.0
    brier_sum = 0.0
    rps_sum = 0.0

    details = []

    # Per-outcome tracking
    outcome_correct = {"W": 0, "D": 0, "L": 0}
    outcome_total = {"W": 0, "D": 0, "L": 0}

    for i, (pred, result) in enumerate(zip(predictions, actual_results)):
        actual = result["actual"]
        actual_idx = label_map[actual]

        pred_probs = [pred["ph"], pred["pd"], pred["pa"]]
        pred_idx = int(np.argmax(pred_probs))
        pred_result = inv_map[pred_idx]

        is_correct = (pred_result == actual)
        if is_correct:
            correct += 1

        # Track per-outcome
        outcome_total[actual] += 1
        if is_correct:
            outcome_correct[actual] += 1

        # Log-loss
        clipped = max(pred_probs[actual_idx], 0.001)
        log_loss_sum += -math.log(clipped)

        # Brier score
        actual_onehot = [0.0, 0.0, 0.0]
        actual_onehot[actual_idx] = 1.0
        brier_sum += sum((p - a) ** 2 for p, a in zip(pred_probs, actual_onehot))

        # RPS (Ranked Probability Score)
        y = actual_onehot
        rps_sum += 0.5 * ((pred_probs[0] - y[0]) ** 2 +
                          (pred_probs[0] + pred_probs[1] - y[0] - y[1]) ** 2)

        details.append({
            "match": f"{result.get('home', '?')} vs {result.get('away', '?')}",
            "actual": actual,
            "predicted": pred_result,
            "correct": is_correct,
            "confidence": round(max(pred_probs), 4),
            "probs": {
                "home_win": round(pred["ph"], 4),
                "draw": round(pred["pd"], 4),
                "away_win": round(pred["pa"], 4),
            },
        })

    accuracy = correct / n if n > 0 else 0.0

    return {
        "total_matches": n,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_log_loss": round(log_loss_sum / n, 4) if n else 0,
        "avg_brier_score": round(brier_sum / n, 4) if n else 0,
        "avg_rps": round(rps_sum / n, 4) if n else 0,
        "outcome_accuracy": {
            k: round(outcome_correct[k] / max(outcome_total[k], 1), 4)
            for k in ["W", "D", "L"]
        },
        "outcome_counts": {
            k: {"actual": outcome_total[k], "correct": outcome_correct[k]}
            for k in ["W", "D", "L"]
        },
        "details": details,
    }


# ─────────────────────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    """
    Main: predict all group matches, evaluate on completed ones.
    """
    # Generate all group fixtures
    fixtures = []
    for group_name, teams in GROUPS.items():
        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                fixtures.append({"group": group_name, "home": home, "away": away})

    # Predict all fixtures
    all_predictions = []
    for f in fixtures:
        pred = dixon_coles_match_prob(f["home"], f["away"])
        pred["group"] = f["group"]
        pred["home"] = f["home"]
        pred["away"] = f["away"]
        all_predictions.append(pred)

    # Evaluate on completed matches
    completed_preds = []
    completed_results = []

    for m in COMPLETED_MATCHES:
        pred = dixon_coles_match_prob(m["home"], m["away"])
        completed_preds.append(pred)

        hs, as_ = m["home_score"], m["away_score"]
        if hs > as_:
            actual = "W"
        elif hs < as_:
            actual = "L"
        else:
            actual = "D"

        completed_results.append({
            "home": m["home"],
            "away": m["away"],
            "actual": actual,
        })

    report = evaluate_predictions(completed_preds, completed_results)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Dixon-Coles Model Evaluation (Hicruben calibrated Elo)")
    print(f"{'='*60}")
    print(f"  Matches evaluated:  {report['total_matches']}")
    print(f"  Accuracy:           {report['correct']}/{report['total_matches']} ({report['accuracy']*100:.1f}%)")
    print(f"  Log-loss:           {report['avg_log_loss']:.4f}")
    print(f"  Brier score:        {report['avg_brier_score']:.4f}")
    print(f"  RPS:                {report['avg_rps']:.4f}")
    print(f"\n  Per-outcome accuracy:")
    for k in ["W", "D", "L"]:
        c = report['outcome_counts'][k]
        label = {"W": "Home win", "D": "Draw", "L": "Away win"}[k]
        pct = report['outcome_accuracy'][k] * 100
        print(f"    {label:12s}: {c['correct']}/{c['actual']} ({pct:.0f}%)")

    print(f"\n  Match-by-match details:")
    print(f"  {'Match':40s} {'Actual':6s} {'Pred':6s} {'Prob':6s} {'':3s}")
    print(f"  {'-'*65}")
    for d in report["details"]:
        mark = "✓" if d["correct"] else "✗"
        print(f"  {d['match']:40s} {d['actual']:6s} {d['predicted']:6s} {d['confidence']*100:5.1f}% {mark}")

    # Print upcoming predictions
    print(f"\n{'='*60}")
    print(f"  Upcoming Match Predictions")
    print(f"{'='*60}")

    # Find matches not yet played
    completed_set = set()
    for m in COMPLETED_MATCHES:
        completed_set.add((m["home"], m["away"]))

    for p in all_predictions:
        if (p["home"], p["away"]) not in completed_set:
            probs = [p["ph"], p["pd"], p["pa"]]
            max_idx = int(np.argmax(probs))
            result = ["W", "D", "L"][max_idx]
            result_label = {"W": f"{p['home']}胜", "D": "平局", "L": f"{p['away']}胜"}[result]
            print(f"  {p['home']:20s} vs {p['away']:20s} → {result_label:20s} "
                  f"(H:{p['ph']*100:.0f}% D:{p['pd']*100:.0f}% A:{p['pa']*100:.0f}% | "
                  f"xG {p['home_xg']:.2f}-{p['away_xg']:.2f})")


if __name__ == "__main__":
    main()
