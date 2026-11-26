# Unsupervised health checks and synthetic benchmarks instead of hand-labeled test dev sets

Competition rules prohibit manual inspection, filling, or editing of hidden test answers, while leaderboard submissions are rate-limited. We decided never to hand-label answers from `tests.xlsx` for local evaluation; instead, offline iteration relies on synthetic/public table unit benchmarks plus automated unsupervised health checks (strict format/JSON validity, row/column subtotal consistency, uncertainty rate, and multi-sample self-consistency), validated against periodic leaderboard submissions.

## Considered Options

- **Synthetic benchmarks + unsupervised health checks + leaderboard (chosen)**: Eliminates anti-cheating compliance risk and overfitting to hand-inspected test items while still catching regressions automatically.
- **Hand-labeling a subset of `tests.xlsx` as a local dev set**: Provides direct local accuracy scores but risks breaching competition anti-cheating rules.
