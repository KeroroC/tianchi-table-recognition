# Page-atomic stage-1 extraction without physical cross-page table merging

Multi-page PDFs can contain both multiple independent tables and tables that continue across consecutive pages. We decided to keep stage-1 extraction strictly page-atomic—producing one cached `PageArtifact` per page—and delegate cross-page continuation linking to the stage-2 locator and solver rather than physically merging tables across pages during stage 1.

## Considered Options

- **Page-atomic caching + stage-2 multi-page assembly (chosen)**: Keeps page extraction parallelizable, independently retryable, and safely cacheable, while avoiding irreversible merge errors when adjacent pages hold separate tables.
- **Stage-1 physical cross-page table merging**: Simplifies downstream single-table lookup when accurate, but couples caching across pages and makes false merges impossible to undo without re-running stage 1.
