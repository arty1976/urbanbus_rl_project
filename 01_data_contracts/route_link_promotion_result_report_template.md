# Route Link Promotion Result Report Template

이 문서는 `03_validation_queries/route_link_promotion_readiness.sql` 실행 결과를 붙여서 최종 판정(승인 / 보류 / 차단)을 기록하는 템플릿입니다.

관련 문서:
- 검증 SQL(SQL=Structured Query Language=구조화 질의 언어): `03_validation_queries/route_link_promotion_readiness.sql`
- 판독 기준표: `01_data_contracts/route_link_promotion_result_guide.md`

---

## 1. 실행 정보

| 항목 | 값 |
|---|---|
| 실행 일시 | |
| 실행자 | |
| DB 대상 | |
| 대상 테이블 | `public.stg_daegu_route_links_api`, `public.route_link_sequence`, `public.stg_daegu_routes` |
| 기준 문서 | `route_link_promotion_readiness.sql`, `route_link_promotion_result_guide.md` |
| 비고 | |

## 2. 검증 전제

- 원천 기준 전체 노선 수: `238`
- 예외 노선 수: `4`
- 예외 노선 목록:
  - `4040006020`
  - `4040006021`
  - `4040007001`
  - `4040007009`
- 유효 검증 대상 노선 수: `234`
- legacy sample route: `1000`

---

## 3. Summary 결과

### 3-1. 기대값

| section | 기대값 |
|---|---:|
| `source_routes_total` | 238 |
| `excluded_route_count` | 4 |
| `eligible_source_routes` | 234 |
| `staging_effective_distinct_routes` | 234 |
| `promoted_distinct_routes_all` | 234 |
| `promoted_effective_distinct_routes` | 234 |
| `eligible_route_count_status` | PASS |

### 3-2. 실제값 기록

| section | 기대값 | 실제값 | 판정 |
|---|---:|---:|---|
| `source_routes_total` | 238 |  |  |
| `excluded_route_count` | 4 |  |  |
| `eligible_source_routes` | 234 |  |  |
| `staging_effective_distinct_routes` | 234 |  |  |
| `promoted_distinct_routes_all` | 234 |  |  |
| `promoted_effective_distinct_routes` | 234 |  |  |
| `eligible_route_count_status` | PASS |  |  |

### 3-3. Summary 원문 붙여넣기

```text
[여기에 summary 결과 원문을 그대로 붙여넣는다]
```

---

## 4. 상세 결과 판정표

| check_name | 기대 상태 | 실제 행 수 | 판정 | 비고 |
|---|---|---:|---|---|
| `missing_route_in_staging` | 0행 |  |  |  |
| `unexpected_route_in_staging` | 0행이 이상적 (`1000`만 있으면 WARN) |  |  |  |
| `staging_dup_natural_key` | 0행 |  |  |  |
| `unexpected_route_in_promoted` | 0행 |  |  |  |
| `promoted_dup_natural_key` | 0행 |  |  |  |
| `count_mismatch_by_route_dir` | 0행 |  |  |  |
| `key_gap_between_staging_and_promoted` | 0행 |  |  |  |
| `promoted_continuity_break` | 0행 |  |  |  |
| `link_id_collision_between_staging_and_promoted` | 0행 |  |  |  |

### 4-1. 상세 결과 원문 붙여넣기

#### `missing_route_in_staging`
```text
[결과 없으면 "0 rows" 또는 실행 도구 표시 그대로 기록]
```

#### `unexpected_route_in_staging`
```text
[결과 없으면 "0 rows" 또는 실행 도구 표시 그대로 기록]
```

#### `staging_dup_natural_key`
```text
[결과 없으면 "0 rows" 또는 실행 도구 표시 그대로 기록]
```

#### `unexpected_route_in_promoted`
```text
[결과 없으면 "0 rows" 또는 실행 도구 표시 그대로 기록]
```

#### `promoted_dup_natural_key`
```text
[결과 없으면 "0 rows" 또는 실행 도구 표시 그대로 기록]
```

#### `count_mismatch_by_route_dir`
```text
[결과 없으면 "0 rows" 또는 실행 도구 표시 그대로 기록]
```

#### `key_gap_between_staging_and_promoted`
```text
[결과 없으면 "0 rows" 또는 실행 도구 표시 그대로 기록]
```

#### `promoted_continuity_break`
```text
[결과 없으면 "0 rows" 또는 실행 도구 표시 그대로 기록]
```

#### `link_id_collision_between_staging_and_promoted`
```text
[결과 없으면 "0 rows" 또는 실행 도구 표시 그대로 기록]
```

---

## 5. 최종 판정

### 최종 상태
- [ ] 승인
- [ ] 보류
- [ ] 차단

### 판정 사유
- 
- 
- 

### 판정 규칙 적용 메모
- **승인**: summary 기대값 일치 + promoted 오염 없음 + 상세 FAIL 항목 전부 0행
- **보류**: promoted 는 정상이나 staging 에 legacy sample `1000` 만 남아 있는 경우
- **차단**: promoted 오염, 자연키 중복, 건수 불일치, key gap, 연속성 단절, `link_id` 충돌 중 하나라도 존재

---

## 6. 후속 조치

### 승인 시
- route_link_sequence 승격 결과를 기준본으로 채택한다.
- 후속 분석/모델링 단계로 진행한다.

### 보류 시
- staging 청소 여부를 결정한다.
- legacy sample `1000` 제거 후 재검증 여부를 기록한다.

### 차단 시
- 아래 항목을 우선 수정 대상으로 기록한다.
  - 
  - 
  - 
- 수정 후 `route_link_promotion_readiness.sql` 재실행한다.

---

## 7. 승인 기록

| 항목 | 값 |
|---|---|
| 작성자 | |
| 검토자 | |
| 승인 여부 | |
| 승인 일시 | |
| 비고 | |
