# graph_edge_master — STOP_TO_STOP 복원 패키지

**작성일**: 2026-04-19
**프로젝트 경로 (canonical)**: `C:\Users\ryujo\urbanbus_rl_project\`
**본 패키지 경로**: `C:\Users\ryujo\urbanbus_rl_project\graph_edge_master_pkg\`
**목적**: GATv2 기반 RL 용 static graph 의 `graph_edge_master` 테이블에
`STOP_TO_STOP` 엣지를 `route_link_sequence` + `dim_stop` 로부터 생성/복구.

> **USB 백업 (선택)**: `robocopy C:\Users\ryujo\urbanbus_rl_project D:\urbanbus_rl_project /E /XO`

---

## 0. 전제: 확인된 실제 스키마

### `public.route_link_sequence`
| column | type | note |
|---|---|---|
| `route_id` | text NOT NULL | |
| `move_dir_code` | **text** NOT NULL | `::integer` 캐스트 예정 |
| `link_seq` | integer NOT NULL | PK 일부 |
| `link_id` | text NOT NULL | |
| `st_node_id` | text | **stop_id 로 가정** (inner join 으로 검증) |
| `ed_node_id` | text | **stop_id 로 가정** |
| `gis_dist` | numeric | 참고용, 본 로드에서는 geom_5187 기반으로 재계산 |
| PK | `(route_id, move_dir_code, link_seq)` | |

### `public.dim_stop`
| column | type | note |
|---|---|---|
| `stop_id` | varchar(50) PK | |
| `longitude` / `latitude` | numeric(12,8) | |
| **`geom_5187`** | geometry(Point,5187) | 미터 단위 CRS, `ST_Distance` 에 사용 |
| `geom_4326` | geometry(Point,4326) | |

### `public.graph_edge_master` (기존, 2026-04-10 생성)
`distance_m`(numeric), `time_sec`(numeric) 컬럼을 이미 보유 → 사용자 요청 컬럼명
(`edge_distance_m`, `base_travel_time_min`)과 네임 불일치. 본 패키지는 **기존
컬럼을 그대로 사용** 하고, 누락된 `is_primary_edge`(boolean) / `edge_rank`(smallint)만
`ALTER TABLE ADD COLUMN IF NOT EXISTS` 로 추가함.

---

## 1. 파일 구성

| # | 파일 | 역할 |
|---|---|---|
| 1 | `01_alter_graph_edge_master.sql` | 누락 컬럼/인덱스 추가 (idempotent) |
| 2 | `02_load_graph_edge_master_from_route_link_sequence.sql` | STOP_TO_STOP 엣지 생성/UPSERT, 중복 pair rank 부여, stale 엣지 비활성화 |
| 3 | `03_graph_edge_master_readiness.sql` | R0~R12 검증 쿼리 |
| 4 | `run_graph_edge_master.ps1` | 01→02→03 순차 실행, 타임스탬프 로그 저장 |
| 5 | `README.md` | 본 문서 |

---

## 2. 실행 방법

### 2-A. PowerShell 일괄 실행 (권장)

```powershell
# 1) 환경변수 세팅 (필요 시)
$env:PGHOST     = "localhost"
$env:PGPORT     = "5432"
$env:PGUSER     = "postgres"
$env:PGDATABASE = "urbanbus"
$env:PGPASSWORD = "your_password"   # 또는 pgpass.conf 사용

# 2) 실행
cd C:\Users\ryujo\urbanbus_rl_project\graph_edge_master_pkg
.\run_graph_edge_master.ps1
```

실행 후 `logs_YYYYMMDD_HHMMSS\` 아래에 단계별 `.log` 파일이 생성됩니다.

#### 2-A-1. psql 이 PATH 에 없을 때

`'psql' 용어가 ... 인식되지 않습니다` 오류가 나면 PowerShell 스크립트가 PostgreSQL 기본 설치 경로(`C:\Program Files\PostgreSQL\<버전>\bin\psql.exe`)를 자동 탐지하지만, 커스텀 경로에 설치된 경우 아래처럼 명시 지정:

```powershell
# 본 세션 한정
$env:PSQL_EXE = "C:\Program Files\PostgreSQL\16\bin\psql.exe"
.\run_graph_edge_master.ps1

# 또는 PATH 에 영구 추가 (관리자 PowerShell)
[Environment]::SetEnvironmentVariable(
    "Path",
    $env:Path + ";C:\Program Files\PostgreSQL\16\bin",
    "Machine"
)
```

설치된 PostgreSQL 버전/경로를 먼저 확인:
```powershell
Get-ChildItem "C:\Program Files\PostgreSQL" -Directory | Select-Object Name
```

### 2-B. 개별 psql 실행

```powershell
psql -h localhost -U postgres -d urbanbus -v ON_ERROR_STOP=1 -f 01_alter_graph_edge_master.sql
psql -h localhost -U postgres -d urbanbus -v ON_ERROR_STOP=1 -f 02_load_graph_edge_master_from_route_link_sequence.sql
psql -h localhost -U postgres -d urbanbus -v ON_ERROR_STOP=1 -f 03_graph_edge_master_readiness.sql | Tee-Object readiness.log
```

---

## 3. 원칙 및 정책 (user-approved 2026-04-19)

1. **GATv2 기반 static graph** — 본 엣지는 학습 중 변경되지 않음.
2. **Stop-to-stop directed edge only** — 버스 node, dynamic edge, 혼잡도 엣지는 본 패키지 대상 아님.
3. **거리 계산 (v2)**: leg-aggregation 단계에서 누적한 `gis_dist` 합 우선,
   값이 0/NULL 이면 `dim_stop.geom_5187` 의 `ST_Distance` (미터 단위) fallback. → `distance_m` 컬럼.
4. **기본 통행시간**: 평균속도 **20 km/h** 가정한 정적 추정치.
   ```
   time_sec = (distance_m / 1000.0) / 20.0 * 3600.0
   ```
   → `base_travel_time_min` 환산은 `time_sec / 60.0`.
5. **자기루프 금지** (`src_stop = dst_stop` leg 제외).
6. **Orphan 금지** (`dim_stop` 매칭 실패 stop 은 엣지 생성 제외).
7. **중복 pair 처리**: 동일 `(src_stop, dst_stop)` 에 여러 노선이 관여하면
   - `distance_m ASC` 로 정렬해 `edge_rank` (1,2,3…) 부여
   - `edge_rank = 1` → `is_primary_edge = true`
   - tie-break: `route_id, move_dir_code, link_seq`.
8. **route_id / move_dir_code 보존** — `edge_uid` 에 포함해 재학습/분석 시 역추적 가능.
9. **delta_t_hr_raw 저장 금지** — 시간 차원 피처는 `graph_state_timeslice` 전용.
10. **Idempotent** — 재실행 시 `ON CONFLICT ... DO UPDATE` 로 안전.
11. **Stale 처리** — 소스에서 사라진 엣지는 `is_active = false` 로 표시 (DELETE 아님).
12. **Leg-aggregation (v2 신규)** — `route_link_sequence` 는 정류장 + 도로/교차로
    intermediate node 가 혼합된 시퀀스이므로, 두 끝이 모두 stop 인 link 만 적재
    하면 coverage 가 낮아진다 (v1 측정 22.84%). v2 부터는 `(route_id, move_dir_code)`
    별로 `link_seq` 를 순회하면서 `st_is_stop=TRUE` 마다 `leg_id` 를 증가시켜,
    "한 stop 출발 ~ 다음 stop 도착" 까지의 모든 link 를 하나의 STOP_TO_STOP
    엣지로 접는다. intermediate node 자체는 `graph_edge_master` 에 저장하지 않는다.

---

## 4. edge_uid 포맷

```
STOP_TO_STOP:<route_id>:<move_dir_code>:<src_stop_id>-><dst_stop_id>
```

예시:
```
STOP_TO_STOP:R1000:0:702341->702342
STOP_TO_STOP:R1000:1:702342->702341
```

- `src_node_uid` = `STOP:<src_stop_id>` / `dst_node_uid` = `STOP:<dst_stop_id>`
- `edge_type` = `'STOP_TO_STOP'`
- `is_bidirectional` = `false` (방향성 엣지)
- **v2 주의**: 한 leg 는 복수의 link 를 접어 만든 엣지이지만, 해당 엣지에
  보관되는 `link_id` / `link_seq` 는 **leg 의 첫 link** 값이다 (역추적 시작점).
  엣지 고유성은 `(route_id, move_dir_code, src_stop, dst_stop)` 로 확보되며,
  동일 노선이 같은 stop pair 를 여러 leg 로 방문해도 (회차·순환) leg 중
  `link_seq` 가 가장 이른 1건만 유지한다 (STAGE 5 dedup).

기존 spec 의 `LINK_TO_LINK:R1000:0:12:8800123->8800456` 패턴을 따릅니다 (단, STOP_TO_STOP 은 link_seq 대신 src/dst 자체가 고유성 확보).

---

## 5. Readiness 통과 기준

| ID | 체크 | 기대값 |
|---|---|---|
| R1 | `stop_to_stop_total` | > 0 |
| R2 | `primary_edges` | > 0 AND ≤ total |
| R3 | `self_loop_cnt` | **0** |
| R4 | `orphan_src`, `orphan_dst` | **0, 0** |
| R5 | duplicate primary rows | **0** |
| R6 | `null_distance`, `null_time`, `zero_distance` | **0, 0, 0** |
| R7 | `max_dist_m` | < 20000 (대구시 내 상식적 상한) |
| R10 | `coverage_pct` | 95% 이상 권장 (낮으면 R11 의 orphan 샘플 조사) |

---

## 6. 사후 작업 (패키지 실행 후)

1. `logs_*/03_readiness.log` 의 핵심 숫자를 `project_log.md` 에 `## 2026-04-19 graph_edge_master STOP_TO_STOP 복원` 섹션으로 기록.
2. R10 leg-coverage_pct < 95% 이면 02_load 의 NOTICE 값 (`leg edges aggregated`,
   `tmp_edge_src rows`, `tmp_edge_dedup rows`) 과 R11 의 non-stop 샘플을 비교해
   (a) 특정 노선의 terminal leg 가 닫히지 않은 경우(마지막 link 의 ed_is_stop=FALSE)
   (b) `dim_stop.geom_5187` NULL stop 누락 (STAGE 4 의 inner join 로 제외됨)
   중 원인 분류.
3. R3/R4/R5/R6 중 하나라도 0 이 아니면 반드시 원인 분석 후 재로드.
4. 이후 단계: `graph_state_timeslice` 에 `node_uid = 'STOP:<stop_id>'` 기준으로
   state features 를 붙일 때 본 엣지들이 GATv2 message-passing 구조로 사용됨.
5. (선택) USB 백업:
   ```powershell
   robocopy "C:\Users\ryujo\urbanbus_rl_project" "D:\urbanbus_rl_project" /E /XO /R:2 /W:5
   ```

---

## 7. 변경 이력

| 일자 | 내용 |
|---|---|
| 2026-04-19 | 초기 작성, 스키마 불일치 3건 (distance_m / time_sec / primary+rank 미보유) 확인 후 ALTER 방식으로 해결 |
| 2026-04-19 | 작업 폴더 C:\work\urbanbus_cowork 로 확정 (Cowork USB 비호환 우회) |
| 2026-04-19 | 프로젝트 canonical 경로 C:\Users\ryujo\urbanbus_rl_project 로 재확정. USB D:\ 는 백업 전용 |
| 2026-04-19 | **02_load v2 배포 — leg-aggregation 방식** (D1~D3 진단 결과 확정된 가설 D 반영): route_link_sequence 가 stop + intermediate node 혼합 시퀀스임이 확인됨. v1 은 두 끝이 stop 인 link 만 적재 → coverage 22.84%. v2 는 `(route_id, move_dir_code)` 별 leg_id window 집계로 "연속된 두 stop 사이" 모든 link 를 하나의 STOP_TO_STOP 엣지로 접음. edge_uid 단위 dedup 추가 (STAGE 5). intermediate node 는 graph_edge_master 에 저장하지 않음. 03_readiness R10 coverage 분모를 "source link rows" → "leg-aggregation distinct legs" 로 변경. R11 도 "매칭 실패" → "intermediate node 샘플" 로 semantics 수정. |
