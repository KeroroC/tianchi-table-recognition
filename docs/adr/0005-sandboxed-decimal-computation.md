# Sandboxed Decimal Python execution for thinking questions

Aggregation, ratio, and multi-step arithmetic in `thinking` questions (216 items) frequently fail under pure LLM mental math or accumulate floating-point rounding drift. We decided to have the text reasoning model generate constrained Python calculation scripts using `decimal.Decimal` and execute them in an isolated, no-IO, time-bounded subprocess rather than relying on fixed JSON operator tools or unassisted LLM arithmetic.

## Considered Options

- **Constrained Python + `Decimal` subprocess sandbox (chosen)**: Handles arbitrary multi-step financial logic and conditional filters accurately without floating-point drift or host side effects, with fallback to direct reasoning on script failure.
- **Fixed JSON aggregation operators (`sum`/`avg`/`diff`)**: Safer surface area, but too rigid for compound financial expressions, multi-condition filters, and chained ratios.
- **Direct LLM mental math**: Simplest, but prone to silent arithmetic errors on large multi-row tables.
