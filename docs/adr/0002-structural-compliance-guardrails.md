# Structural compliance guardrails for model gateway and filename stripping

Competition 532510 mandates exclusive use of Alibaba Cloud Tongyi Qianwen / DashScope / Bailian services in the final pipeline and prohibits exploiting filenames or non-table external signals. We decided to enforce both rules structurally in code: all model/OCR traffic flows through a single gateway with a strict compliant-provider allowlist and audit log, and inference context builders strip raw filenames in favor of opaque `DocumentKey` identifiers.

## Consequences

- Debugging model inputs requires mapping `DocumentKey` back to the source document via run metadata rather than reading filenames in prompts.
- Any attempt to invoke a non-compliant model or local private model fails fast at the gateway boundary rather than silently entering a submission.
