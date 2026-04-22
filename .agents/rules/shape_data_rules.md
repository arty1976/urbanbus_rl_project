---
trigger: always_on
---

# Shape and data contract rules

This project uses staged promotion from raw data to staging tables and then to analytical tables.

Core rules:
1. Never join by Korean stop name alone.
2. Prefer stable IDs over names whenever possible.
3. Before proposing promotion to dim_stop, dim_route, or bridge_route_stop, always record:
   - layer name
   - row count
   - geometry type
   - candidate key columns
   - null risks
   - duplicate risks
4. Do not infer stop_id, route_id, node_id, or link_id unless the source columns clearly support that interpretation.
5. If candidate IDs are ambiguous, explicitly say they are ambiguous.
6. Promotion is allowed only after:
   - schema profiling is complete
   - key candidates are documented
   - row-level risks are stated
   - join assumptions are written down
7. Never overwrite raw source files.
8. All profiling results must be written to markdown under 01_data_contracts/.

Output style rules:
- Be precise and skeptical.
- Separate observations from interpretations.
- Separate confirmed IDs from candidate IDs.
- When uncertain, say uncertain.