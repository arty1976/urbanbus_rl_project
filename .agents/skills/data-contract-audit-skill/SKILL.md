---
name: data-contract-audit-skill
description: Audit data structure, candidate keys, null semantics, duplicate risks, and promotion readiness for transport datasets.
---

# Purpose

Use this skill when inspecting CSV (Comma-Separated Values=쉼표 구분 텍스트 파일), SHP (Shapefile=공간 벡터 파일 형식), DBF (Database File=셰이프파일 속성 테이블 파일), or staging tables before promoting them into project data contracts.

# What this skill must do

For each dataset or layer:

1. Identify the physical source
   - file path
   - file type
   - layer name if spatial
2. Summarize schema
   - column names
   - probable meaning of each important column
   - likely identifier columns
3. Evaluate key quality
   - primary key candidate
   - composite key candidate
   - duplicate risk
   - null risk
4. Evaluate join readiness
   - can it join to stop
   - can it join to route
   - can it join to node
   - can it join to link
   - what assumptions are required
5. Evaluate promotion readiness
   - suitable for dim_stop
   - suitable for dim_route
   - suitable for bridge_route_stop
   - not ready yet
6. Record the result in markdown under 01_data_contracts/

# Required output structure

Always produce these sections:

## Source
## Schema summary
## Candidate key columns
## Null and duplicate risks
## Join possibilities
## Promotion assessment
## Open questions

# Important rules

- Never join by stop name only.
- Never pretend a field is stop_id or route_id unless the evidence is strong.
- If the meaning of a field is unclear, say unclear.
- Distinguish confirmed facts from likely interpretations.
- If command execution is needed, show the exact command first and ask for review before execution.

# Promotion criteria

A dataset is promotion-ready only if:
- key candidates are documented
- duplicate behavior is explained
- null behavior is explained
- join targets are explicit
- promotion target is justified