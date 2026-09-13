# Decision Log

1. **Use Customer Support on Twitter as the primary dataset.**
   It is explicitly requested by the assignment and contains real, noisy support interactions.

2. **Focus on one brand.**
   This keeps the scope small enough to evaluate rigorously.

3. **Start with AppleSupport as the candidate brand.**
   It has a large number of linked customer/reply interactions; the final selection should be confirmed from the actual data.

4. **Do not use the entire dataset for experiments.**
   A reproducible subsample is enough and keeps runtime below the assignment's target.

5. **Build conversation-level splits.**
   This reduces leakage between retrieval/training data and evaluation data.

6. **Derive intents from the selected brand's data.**
   Generic sentiment labels such as positive/negative are not sufficient for this assignment.

7. **Use a traditional ML baseline.**
   TF-IDF + Logistic Regression provides a simple, understandable benchmark.

8. **Use historical support cases as retrieval evidence.**
   The reply generator should be grounded in how the brand historically handled similar issues.

9. **Use conservative escalation.**
   Low-confidence, high-risk, ambiguous, or unsupported cases should go to a human.

10. **Evaluate the judge against humans.**
    An LLM judge should not be treated as ground truth without calibration.

11. **Report failure modes, not only aggregate metrics.**
    A single headline number can hide poor performance on difficult cases.

12. **Avoid production integrations.**
    Twitter posting, CRM actions, authentication, and other production features are outside the take-home scope.
