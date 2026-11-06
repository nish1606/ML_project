# Group 4 – Relational-Rule Learning and Explanation-Based Learning (EBL) Comparison

## Title
Relational-Rule Learning using a FOIL-style Algorithm and Explanation-Based Learning (EBL)

## Aim
To understand and demonstrate:
- Relational learning using first-order logic.
- How a FOIL-style algorithm learns relational rules by adding literals incrementally.
- How domain theory supports Explanation-Based Learning.
- The differences between FOIL and EBL in learning process, generalization, and knowledge requirements.

## Theory

### Relational learning and first-order logic
Relational learning discovers patterns that involve objects and relationships among them, not just fixed attribute vectors. First-order logic represents such knowledge using predicates, variables, and rules. In this experiment, `parent(X,Y)` is a background predicate and `ancestor(X,Y)` is the target relation.

### FOIL-style learning
FOIL (First Order Inductive Learner) learns logical rules from positive and negative examples. It starts with a general rule head and adds body literals one by one. Each literal is chosen to improve discrimination: keep positive examples covered while reducing negative examples covered. In this lab, a simplified FOIL-style learner searches a small literal language and uses a gain/coverage score to select literals.

### Explanation-Based Learning (EBL)
EBL uses an existing domain theory to explain why a training example satisfies a target concept. The explanation (a proof trace) is then generalized by replacing specific constants with variables, producing a reusable rule. EBL can generalize from few examples when the domain theory is correct, but it does not invent knowledge outside that theory.

## Algorithm

### FOIL-style algorithm (simplified)
1. Input: positive examples, negative examples, background facts (`parent`), target `ancestor(X,Y)`.
2. Initialize an empty rule set.
3. While uncovered positive examples remain:
   - Start a new rule: `ancestor(X,Y) :-`.
   - While the rule covers negative examples:
     - Generate candidate literals (`parent(X,Y)`, `parent(X,Z)`, `parent(Z,Y)`, recursive `ancestor` literals when appropriate).
     - Score each literal by its effect on positive/negative coverage (FOIL-style gain).
     - Add the best literal to the rule and record the step.
     - Stop if positives are lost or no useful literal exists.
   - Add the rule if it covers positives and excludes negatives.
   - Remove positives covered by the rule.
4. Output learned rules and trace.

### EBL algorithm
1. Input: training example `ancestor(x,y)`, domain theory:
   - `ancestor(X,Y) :- parent(X,Y).`
   - `ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y).`
2. Explain the example by proving it from domain theory.
3. Record the proof trace (rules used, bindings, facts/subproofs).
4. Generalize the explanation by keeping the explanatory rule structure.
5. Apply the generalized rule/domain theory to unseen examples.

## Implementation
The implementation is in `group4_foil_ebl.py` and uses:
- Python data structures (`set`, `list`, `tuple`, `dict`) for facts, rules, and substitutions.
- `pandas` for facts, examples, trace, metrics, and comparison tables.
- `matplotlib` for visualization.

Sections in code:
1. Import libraries.
2. Create synthetic relational data (`PARENT_FACTS`, `PEOPLE`).
3. Generate positive and negative examples from graph reachability.
4. Implement FOIL-style learning (`FOILLearner`).
5. Implement EBL (`EBLLearner`).
6. Test both approaches on held-out examples.
7. Compare results using accuracy, precision, recall, and F1 computed from predictions.
8. Visualize metric comparison and FOIL literal-selection effect (`group4_foil_ebl_results.png`).

The FOIL rules are learned by the algorithm from data; they are not hardcoded as the result. EBL is explicitly separated from ordinary supervised learning because it uses domain theory to explain/generalize a concept.

## Results
Running the script produces:
- Sample relational facts (parent relationships).
- Positive and negative ancestor examples.
- FOIL literal-selection steps with `pos_before/neg_before` and `pos_after/neg_after`.
- Final learned FOIL-style rules.
- EBL explanation trace for a training example (e.g., `ancestor(Tom,Alice)`).
- Predictions on unseen examples.
- Accuracy/precision/recall/F1 computed from actual program outputs.
- Comparison table and visualization image.

To generate actual outputs:
```bash
python group4_foil_ebl.py
```

## Discussion
- **Why FOIL adds literals:** A general rule like `ancestor(X,Y)` can cover many pairs; literals constrain it. `parent(X,Y)` captures direct ancestry, while adding `parent(X,Z)` and `ancestor(Z,Y)` captures transitive ancestry. Each step is selected to improve separation between positive and negative examples.
- **How EBL uses domain theory:** EBL proves the training example using known rules and facts, producing an explanation. It then generalizes the explanation into a reusable rule.
- **EBL with fewer examples:** If the domain theory is correct, one explained example can justify a broad rule because the proof structure already encodes valid reasoning.
- **FOIL usefulness:** FOIL is valuable when the target rule is unknown and must be induced from labeled examples and background predicates.
- **Incorrect/incomplete domain theory in EBL:** Wrong rules produce wrong explanations; missing rules prevent explanation or limit generalization.
- **Noise/missing facts/negatives in FOIL:** Noise can mislead literal selection, missing facts can hide true positives, and poor negatives can produce overly general or incorrect rules.

## Comparison
| Parameter | FOIL | EBL |
|---|---|---|
| Learning approach | Data-driven relational rule induction by greedy literal addition. | Knowledge-driven explanation and generalization from domain theory. |
| Knowledge required | Background predicates, positive/negative examples. | Correct domain theory and an explainable training example. |
| Role of examples | Central; many examples guide search and evaluation. | Often small; one example may suffice. |
| Role of domain theory | Background support for literals. | Core mechanism for explanation and generalization. |
| Rule generation | Searches candidate literals and selects useful ones. | Derives rule structure from proof/explanation. |
| Generalization | Induced from data patterns and coverage. | Analytical generalization of a valid explanation. |
| Advantages | Can learn unknown rules when theory is weak. | Efficient, interpretable, can generalize from few examples. |
| Limitations | Sensitive to data quality; search cost grows. | Limited by correctness/completeness of domain theory. |

## Limitations
- FOIL depends on suitable background predicates (`parent` here); without useful predicates, it cannot express the target relation.
- FOIL search complexity grows with the number of predicates, variables, constants, and examples.
- This implementation is simplified: it uses a restricted literal language, bounded recursion, and greedy selection rather than full FOIL optimizations.
- EBL depends on accurate domain theory; incorrect theory yields incorrect generalizations.
- EBL has limited ability to discover entirely new knowledge when the domain theory is incomplete.

## Conclusion
The experiment demonstrates two complementary relational learning approaches. FOIL-style learning induces rules from examples by adding literals that separate positive and negative ancestor relationships. EBL uses domain theory to explain a training example and generalize it into a reusable rule. FOIL is useful when rules must be learned from data, while EBL is powerful when reliable domain knowledge is available. The comparison highlights the trade-off between data-driven induction and knowledge-driven explanation.
