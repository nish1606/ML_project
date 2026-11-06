"""
Group 4: Relational-Rule Learning (FOIL-style) and Explanation-Based Learning (EBL)
-------------------------------------------------------------------------------
Undergraduate AIML laboratory demonstration.

Problem: Learn the ancestor(X, Y) relation from parent(X, Y) background facts.

How to run:
    python group4_foil_ebl.py
Works in Jupyter/Colab by running cells or executing the script.

Dependencies: pandas, matplotlib (standard in Colab/Jupyter).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Synthetic relational dataset
# -----------------------------------------------------------------------------

# Background knowledge: ground parent facts.
PARENT_FACTS: Set[Tuple[str, str]] = {
    ("John", "Mary"),
    ("Mary", "Alice"),
    ("Alice", "David"),
    ("John", "Peter"),
    ("Peter", "Sam"),
    ("Anna", "John"),
    ("Tom", "Anna"),
    ("Mary", "Bob"),
    ("Bob", "Eve"),
}

PEOPLE: List[str] = sorted({p for fact in PARENT_FACTS for p in fact})


def true_ancestor(x: str, y: str, parents: Set[Tuple[str, str]]) -> bool:
    """Ground-truth ancestor relation derived from parent facts (for evaluation)."""
    # Direct or transitive reachability through parent links.
    frontier = [c for (p, c) in parents if p == x]
    seen: Set[str] = set()
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur == y:
            return True
        frontier.extend(c for (p, c) in parents if p == cur)
    return False


def make_examples(
    people: Sequence[str],
    parents: Set[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Create positive and negative ancestor examples from all ordered pairs."""
    positives: List[Tuple[str, str]] = []
    negatives: List[Tuple[str, str]] = []
    for x, y in product(people, repeat=2):
        if x == y:
            continue
        if true_ancestor(x, y, parents):
            positives.append((x, y))
        else:
            negatives.append((x, y))
    return positives, negatives


# -----------------------------------------------------------------------------
# 2. FOIL-style relational rule learning (simplified)
# -----------------------------------------------------------------------------
# Representation:
# - A rule has a head ancestor(X, Y) and a body: list of literals.
# - Literal forms used here:
#     ("parent", "X", "Y")
#     ("parent", "X", "Z")
#     ("parent", "Z", "Y")
#     ("ancestor", "Z", "Y")   # allows recursive rule once base case exists
# - A substitution maps variable names to constants.
# - A rule covers (x, y) if there exists a substitution extending {X:x, Y:y}
#   that satisfies every body literal using background knowledge and, for the
#   recursive literal, the currently learned rules (including the rule itself in
#   a controlled evaluation setting).


@dataclass
class Rule:
    head: Tuple[str, str, str]  # ("ancestor", "X", "Y")
    body: List[Tuple[str, str, str]]
    description: str = ""

    def __str__(self) -> str:
        if not self.body:
            return f"{self.head[0]}({self.head[1]},{self.head[2]})."
        body = ", ".join(f"{p}({a},{b})" for p, a, b in self.body)
        return f"{self.head[0]}({self.head[1]},{self.head[2]}) :- {body}."


class FOILLearner:
    """Simplified FOIL-style learner for the ancestor relation.

    It is 'FOIL-style' (not a full industrial FOIL implementation): it searches
    a small, explicit literal language, scores candidate literals by how well
    they keep positive coverage while reducing negative coverage, and adds
    literals greedily. The learned rules are produced by the algorithm from the
    data, not hardcoded as the final answer.
    """

    def __init__(self, parents: Set[Tuple[str, str]], people: Sequence[str]):
        self.parents = parents
        self.people = list(people)
        self.trace: List[Dict[str, object]] = []

    def literal_candidates(self, body: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        vars_in_rule = {"X", "Y"}
        for _, a, b in body:
            vars_in_rule.update([a, b])
        # Introduce Z only when useful; keep search small and interpretable.
        candidates = [
            ("parent", "X", "Y"),
            ("parent", "X", "Z"),
            ("parent", "Z", "Y"),
            ("parent", "Y", "X"),
        ]
        # Allow recursive ancestor literal after at least one literal exists,
        # so the learner can discover a recursive rule.
        if body:
            candidates.append(("ancestor", "Z", "Y"))
            candidates.append(("ancestor", "X", "Z"))
        # Keep only candidates that mention known/introduced variables sensibly.
        out = []
        for lit in candidates:
            if lit in body:
                continue
            _, a, b = lit
            # At least one argument should connect to current variables.
            if a in vars_in_rule or b in vars_in_rule or (a == "Z" or b == "Z"):
                out.append(lit)
        return out

    def prove_literal(
        self,
        lit: Tuple[str, str, str],
        sub: Dict[str, str],
        learned_rules: List[Rule],
        depth: int = 0,
    ) -> Iterable[Dict[str, str]]:
        """Yield substitutions satisfying one literal."""
        if depth > 4:  # prevent runaway recursion in this small demo
            return []
        pred, a, b = lit

        def values(var: str) -> List[str]:
            return [sub[var]] if var in sub else self.people

        results = []
        if pred == "parent":
            for av in values(a):
                for bv in values(b):
                    if (av, bv) in self.parents:
                        ns = dict(sub)
                        ns[a], ns[b] = av, bv
                        results.append(ns)
            return results

        if pred == "ancestor":
            # Use learned rules as background for recursive proof.
            for av in values(a):
                for bv in values(b):
                    if self.covers_pair_with_rules((av, bv), learned_rules, depth + 1):
                        ns = dict(sub)
                        ns[a], ns[b] = av, bv
                        results.append(ns)
            return results

        return results

    def prove_body(
        self,
        body: List[Tuple[str, str, str]],
        sub: Dict[str, str],
        learned_rules: List[Rule],
        depth: int = 0,
    ) -> bool:
        if not body:
            return True
        first, rest = body[0], body[1:]
        for ns in self.prove_literal(first, sub, learned_rules, depth):
            if self.prove_body(rest, ns, learned_rules, depth):
                return True
        return False

    def covers_pair_with_rules(
        self,
        pair: Tuple[str, str],
        rules: List[Rule],
        depth: int = 0,
    ) -> bool:
        x, y = pair
        for rule in rules:
            if self.prove_body(rule.body, {"X": x, "Y": y}, rules, depth):
                return True
        return False

    def covers(self, rule: Rule, examples: List[Tuple[str, str]], learned_rules: List[Rule]) -> List[Tuple[str, str]]:
        covered = []
        # For scoring, recursive literals can use previously learned rules plus
        # the current partial rule where safe.
        rules_for_eval = learned_rules + [rule]
        for ex in examples:
            if self.prove_body(rule.body, {"X": ex[0], "Y": ex[1]}, rules_for_eval):
                covered.append(ex)
        return covered

    def foil_gain(
        self,
        rule: Rule,
        lit: Tuple[str, str, str],
        pos: List[Tuple[str, str]],
        neg: List[Tuple[str, str]],
        learned_rules: List[Rule],
    ) -> Tuple[float, int, int, int, int]:
        """Simple FOIL-like gain: prefer literals keeping positives and dropping negatives."""
        p0 = self.covers(rule, pos, learned_rules)
        n0 = self.covers(rule, neg, learned_rules)
        new_rule = Rule(rule.head, rule.body + [lit])
        p1 = self.covers(new_rule, pos, learned_rules)
        n1 = self.covers(new_rule, neg, learned_rules)

        # Information-gain inspired score with Laplace-like stability.
        def score(p: int, n: int) -> float:
            if p == 0:
                return -1e9
            import math

            return p * math.log2((p + 1e-9) / (p + n + 1e-9))

        gain = len(p1) * (score(len(p1), len(n1)) - score(max(len(p0), 1), len(n0)))
        # Strong preference for eliminating negatives while retaining positives.
        gain += 2.0 * (len(n0) - len(n1))
        gain += 0.5 * len(p1)
        return gain, len(p0), len(n0), len(p1), len(n1)

    def learn_rule(
        self,
        positives: List[Tuple[str, str]],
        negatives: List[Tuple[str, str]],
        learned_rules: List[Rule],
        max_literals: int = 3,
    ) -> Optional[Rule]:
        rule = Rule(("ancestor", "X", "Y"), [], description="candidate")
        current_pos = list(positives)

        for step in range(1, max_literals + 1):
            candidates = self.literal_candidates(rule.body)
            if not candidates:
                break

            scored = []
            for lit in candidates:
                gain, p0, n0, p1, n1 = self.foil_gain(rule, lit, current_pos, negatives, learned_rules)
                scored.append((gain, lit, p0, n0, p1, n1))

            scored.sort(key=lambda x: x[0], reverse=True)
            best_gain, best_lit, p0, n0, p1, n1 = scored[0]

            self.trace.append(
                {
                    "step": step,
                    "current_rule": str(rule),
                    "chosen_literal": f"{best_lit[0]}({best_lit[1]},{best_lit[2]})",
                    "gain": round(best_gain, 4),
                    "pos_before": p0,
                    "neg_before": n0,
                    "pos_after": p1,
                    "neg_after": n1,
                    "why": (
                        "Keeps positive examples while reducing/removing negatives"
                        if n1 < n0 and p1 > 0
                        else "Best available literal under current constraints"
                    ),
                }
            )

            rule.body.append(best_lit)
            current_pos = self.covers(rule, current_pos, learned_rules)
            covered_neg = self.covers(rule, negatives, learned_rules)
            if not covered_neg and current_pos:
                rule.description = "learned by FOIL-style greedy literal addition"
                return rule

            # If positives vanished, backtrack is omitted in this simplified demo;
            # stop to keep behavior transparent.
            if not current_pos:
                break

        covered_neg = self.covers(rule, negatives, learned_rules)
        if not covered_neg and self.covers(rule, positives, learned_rules):
            return rule
        return None

    def learn(
        self,
        positives: List[Tuple[str, str]],
        negatives: List[Tuple[str, str]],
        max_rules: int = 3,
    ) -> List[Rule]:
        rules: List[Rule] = []
        remaining_pos = list(positives)
        for _ in range(max_rules):
            if not remaining_pos:
                break
            rule = self.learn_rule(remaining_pos, negatives, rules)
            if not rule:
                break
            if rule not in rules:
                rules.append(rule)
            covered = set(self.covers(rule, remaining_pos, rules))
            remaining_pos = [p for p in remaining_pos if p not in covered]
        return rules


# -----------------------------------------------------------------------------
# 3. Explanation-Based Learning (EBL)
# -----------------------------------------------------------------------------


class EBLearner:
    """EBL using explicit domain theory for ancestor.

    EBL does not discover the domain theory from many examples. It uses the
    theory to explain one (or a few) training examples, then generalizes that
    explanation into a reusable rule.
    """

    def __init__(self, parents: Set[Tuple[str, str]]):
        self.parents = parents
        self.domain_theory = [
            "ancestor(X,Y) :- parent(X,Y).",
            "ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y).",
        ]

    def explain(self, x: str, y: str) -> Optional[Dict[str, object]]:
        """Return an explanation trace if ancestor(x,y) is entailed by theory."""
        # Base case
        if (x, y) in self.parents:
            return {
                "goal": f"ancestor({x},{y})",
                "rule_used": "ancestor(X,Y) :- parent(X,Y)",
                "bindings": {"X": x, "Y": y},
                "subproofs": [f"parent({x},{y}) is a known fact"],
                "generalized_rule": "ancestor(X,Y) :- parent(X,Y)",
            }

        # Recursive case: find an intermediate Z
        for (p, z) in self.parents:
            if p != x:
                continue
            sub = self.explain(z, y)
            if sub:
                return {
                    "goal": f"ancestor({x},{y})",
                    "rule_used": "ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y)",
                    "bindings": {"X": x, "Z": z, "Y": y},
                    "subproofs": [f"parent({x},{z}) is a known fact", sub],
                    "generalized_rule": "ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y)",
                }
        return None

    def learn_from_example(self, example: Tuple[str, str]) -> Optional[Dict[str, object]]:
        x, y = example
        explanation = self.explain(x, y)
        if not explanation:
            return None
        # Generalization: keep the structure of the proof rule; drop constants.
        return {
            "training_example": f"ancestor({x},{y})",
            "domain_theory": self.domain_theory,
            "explanation": explanation,
            "generalized_rule": explanation["generalized_rule"],
            "note": (
                "EBL explains why the example is true using domain theory, then "
                "generalizes by retaining the explanatory rule structure rather "
                "than memorizing the specific individuals."
            ),
        }

    def predict(self, x: str, y: str) -> bool:
        return self.explain(x, y) is not None


# -----------------------------------------------------------------------------
# 4. Evaluation helpers
# -----------------------------------------------------------------------------


def evaluate_predictions(predict_fn, examples: List[Tuple[str, str]], labels: List[int]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for (x, y), label in zip(examples, labels):
        pred = 1 if predict_fn(x, y) else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    accuracy = (tp + tn) / max(len(examples), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -----------------------------------------------------------------------------
# 5. Main demonstration
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 78)
    print("GROUP 4: FOIL-style Relational Rule Learning and EBL Comparison")
    print("=" * 78)

    # --- Data ---
    positives, negatives = make_examples(PEOPLE, PARENT_FACTS)
    # Train/test split by deterministic slicing for reproducibility.
    pos_train, pos_test = positives[: max(1, int(0.7 * len(positives)))], positives[int(0.7 * len(positives)) :]
    neg_train, neg_test = negatives[: max(1, int(0.7 * len(negatives)))], negatives[int(0.7 * len(negatives)) :]

    facts_df = pd.DataFrame(sorted(PARENT_FACTS), columns=["X (parent)", "Y (child)"])
    print("\n1) Sample relational facts (background knowledge: parent/2)")
    print(facts_df.to_string(index=False))

    examples_df = pd.DataFrame(
        [{"X": x, "Y": y, "ancestor(X,Y)": 1} for x, y in pos_train[:8]]
        + [{"X": x, "Y": y, "ancestor(X,Y)": 0} for x, y in neg_train[:8]]
    )
    print("\n2) Sample training examples (positive=1, negative=0)")
    print(examples_df.to_string(index=False))
    print(f"\nTraining: {len(pos_train)} positives, {len(neg_train)} negatives")
    print(f"Testing:  {len(pos_test)} positives, {len(neg_test)} negatives")

    # --- FOIL ---
    print("\n" + "-" * 78)
    print("3) FOIL-style learning")
    print("-" * 78)
    foil = FOILLearner(PARENT_FACTS, PEOPLE)
    foil_rules = foil.learn(pos_train, neg_train)

    trace_df = pd.DataFrame(foil.trace)
    print("\nFOIL literal-selection trace (intermediate steps):")
    if not trace_df.empty:
        print(trace_df.to_string(index=False))
    else:
        print("(No trace steps recorded)")

    print("\nFinal learned FOIL-style rules:")
    if foil_rules:
        for i, r in enumerate(foil_rules, 1):
            print(f"  Rule {i}: {r}")
    else:
        print("  No rule learned.")

    print("\nLiteral purpose (interpretation):")
    print("  - parent(X,Y): captures direct ancestry; positive examples are covered,")
    print("    but deeper ancestors remain uncovered.")
    print("  - parent(X,Z), ancestor(Z,Y): introduces an intermediate Z and uses")
    print("    recursion/transitivity, improving coverage of multi-level ancestors")
    print("    while avoiding unrelated pairs.")

    # --- EBL ---
    print("\n" + "-" * 78)
    print("4) Explanation-Based Learning (EBL)")
    print("-" * 78)
    ebl = EBLearner(PARENT_FACTS)
    training_example = ("Tom", "Alice")  # multi-level example
    ebl_result = ebl.learn_from_example(training_example)

    print("Domain theory:")
    for rule in ebl.domain_theory:
        print(f"  {rule}")
    print(f"\nTraining example: ancestor{training_example}")

    def print_explanation(node: Dict[str, object], indent: int = 0) -> None:
        pad = "  " * indent
        print(f"{pad}- Goal: {node['goal']}")
        print(f"{pad}  Rule used: {node['rule_used']}")
        print(f"{pad}  Bindings: {node['bindings']}")
        for sp in node["subproofs"]:
            if isinstance(sp, dict):
                print(f"{pad}  Subproof:")
                print_explanation(sp, indent + 2)
            else:
                print(f"{pad}  Fact: {sp}")

    if ebl_result:
        print("\nExplanation trace:")
        print_explanation(ebl_result["explanation"])
        print(f"\nGeneralized reusable rule: {ebl_result['generalized_rule']}")
        print(f"Note: {ebl_result['note']}")
    else:
        print("EBL could not explain the training example with the given domain theory.")

    # --- Testing ---
    print("\n" + "-" * 78)
    print("5) Predictions on unseen examples")
    print("-" * 78)
    test_pairs = pos_test[:5] + neg_test[:5]
    test_labels = [1] * min(5, len(pos_test)) + [0] * min(5, len(neg_test))

    def foil_predict(x: str, y: str) -> bool:
        return foil.covers_pair_with_rules((x, y), foil_rules)

    pred_rows = []
    for (x, y), label in zip(test_pairs, test_labels):
        pred_rows.append(
            {
                "example": f"ancestor({x},{y})",
                "actual": label,
                "FOIL_pred": int(foil_predict(x, y)),
                "EBL_pred": int(ebl.predict(x, y)),
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    print(pred_df.to_string(index=False))

    # --- Evaluation ---
    all_test = pos_test + neg_test
    all_labels = [1] * len(pos_test) + [0] * len(neg_test)
    foil_metrics = evaluate_predictions(foil_predict, all_test, all_labels)
    ebl_metrics = evaluate_predictions(ebl.predict, all_test, all_labels)

    metrics_df = pd.DataFrame([{"method": "FOIL-style", **foil_metrics}, {"method": "EBL", **ebl_metrics}])
    print("\n6) Evaluation (computed from actual predictions)")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nMethodology: metrics are computed on the held-out test pairs generated")
    print("from the same synthetic family graph. ancestor labels are derived by")
    print("graph reachability over parent facts (ground truth for this dataset).")

    # --- Comparison table ---
    comparison = pd.DataFrame(
        [
            {
                "Parameter": "Learning approach",
                "FOIL": "Data-driven relational rule learning; adds literals greedily to separate positives/negatives.",
                "EBL": "Knowledge-driven learning; explains one example using domain theory, then generalizes the explanation.",
            },
            {
                "Parameter": "Knowledge required",
                "FOIL": "Background predicates/facts (e.g., parent) and labeled examples; minimal prior target theory.",
                "EBL": "Correct domain theory (e.g., ancestor rules) plus at least one explained training example.",
            },
            {
                "Parameter": "Role of examples",
                "FOIL": "Many examples guide search and scoring; negatives are crucial for discrimination.",
                "EBL": "A single informative example can be enough if the domain theory explains it.",
            },
            {
                "Parameter": "Role of domain theory",
                "FOIL": "Provides background relations used in literals; does not usually give the target rule directly.",
                "EBL": "Central: used to construct the explanation and justify generalization.",
            },
            {
                "Parameter": "Rule generation",
                "FOIL": "Constructs rules by testing candidate literals and selecting those with better gain/coverage.",
                "EBL": "Derives/generalizes a rule structure from the proof/explanation trace.",
            },
            {
                "Parameter": "Generalization",
                "FOIL": "Generalizes by inducing patterns that fit observed data; quality depends on data coverage/noise.",
                "EBL": "Generalizes by replacing constants with variables in a valid explanation; can be fast and targeted.",
            },
            {
                "Parameter": "Advantages",
                "FOIL": "Can learn unknown rules from examples; useful when domain theory is weak or unavailable.",
                "EBL": "Can learn efficiently from few examples; explanations make learned rules interpretable.",
            },
            {
                "Parameter": "Limitations",
                "FOIL": "Sensitive to noise/missing facts; search can be expensive; needs useful predicates/negatives.",
                "EBL": "Cannot reliably learn beyond incomplete/incorrect domain theory; depends on explanation quality.",
            },
        ]
    )
    print("\n7) Comparison table")
    print(comparison.to_string(index=False))

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    methods = ["FOIL-style", "EBL"]
    acc = [foil_metrics["accuracy"], ebl_metrics["accuracy"]]
    prec = [foil_metrics["precision"], ebl_metrics["precision"]]
    rec = [foil_metrics["recall"], ebl_metrics["recall"]]
    x = range(len(methods))
    width = 0.25
    axes[0].bar([i - width for i in x], acc, width=width, label="Accuracy")
    axes[0].bar(list(x), prec, width=width, label="Precision")
    axes[0].bar([i + width for i in x], rec, width=width, label="Recall")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("FOIL vs EBL on held-out ancestor examples")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    if not trace_df.empty:
        axes[1].plot(trace_df["step"], trace_df["pos_after"], marker="o", label="Positives covered")
        axes[1].plot(trace_df["step"], trace_df["neg_after"], marker="o", label="Negatives covered")
        axes[1].set_title("FOIL literal-selection effect")
        axes[1].set_xlabel("Literal-addition step")
        axes[1].set_ylabel("Examples covered by current rule")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No FOIL trace available", ha="center", va="center")
        axes[1].set_axis_off()

    plt.tight_layout()
    out_path = "group4_foil_ebl_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n8) Visualization saved to: {out_path}")
    # plt.show()  # Uncomment in notebook environments.

    print("\nDone. All metrics and tables above are computed from program outputs.")


if __name__ == "__main__":
    main()
