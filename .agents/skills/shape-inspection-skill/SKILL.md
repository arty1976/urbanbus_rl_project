---
name: shape-inspection-skill
description: Inspect spatial layers such as SHP (Shapefile=공간 벡터 파일 형식) and determine schema, row count, geometry type, and transport key candidates.
---

# Purpose

Use this skill when inspecting spatial transport layers such as:
- bus stop layers
- route layers
- node layers
- link layers

This skill is specialized for identifying:
- stop_id candidates
- route_id candidates
- node_id candidates
- link_id candidates
- geometry type
- promotion suitability

# Execution policy

If terminal or Python (Python=파이썬 프로그래밍 언어) execution is needed:
1. first explain the plan briefly
2. then show the exact command
3. then ask for review
4. only after approval, execute

# Preferred tools

Use one of the following:
- Python with geopandas (geopandas=파이썬 공간 데이터 처리 라이브러리)
- Python with fiona (fiona=파이썬 벡터 데이터 입출력 라이브러리)
- ogrinfo (ogrinfo=공간 데이터 레이어 정보 조회 도구)

# Inspection checklist

For each layer:
1. confirm file exists
2. detect geometry type
3. count rows
4. list columns
5. identify likely identifier columns
6. inspect a few sample records
7. evaluate whether IDs look stable
8. assess whether the layer can support:
   - dim_stop
   - dim_route
   - bridge_route_stop
   - node registry
   - link registry

# Required output structure

Always write results in this format:

## Layer
## File path
## Geometry type
## Row count
## Columns
## Candidate ID fields
## Sample interpretation
## Promotion assessment
## Risks and ambiguities

# Heuristics

Use these heuristics carefully:
- columns containing stop, bstop, station may indicate stop-related IDs
- columns containing route, line may indicate route-related IDs
- columns containing node may indicate node-related IDs
- columns containing link, edge may indicate link-related IDs
- columns ending in id are only candidates, not proof

# Restrictions

- Do not assume semantic meaning from Korean names alone.
- Do not promote a layer unless the key structure is defensible.
- If multiple fields could be the identifier, compare them and document why.