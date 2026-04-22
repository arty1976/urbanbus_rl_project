-- [테이블 간 관계 및 연속성 무결성 검증]

-- 목적: 브릿지 테이블에 연결된 정류장 중 유효하지 않은 고아 정류장 탐색
-- 1. bridge_route_stop.stop_id가 dim_stop에 존재하는지 검사 (고아 정류장 식별)
-- 브릿지 테이블에는 매핑되어 있으나 실제 정류장 마스터 테이블에 존재하지 않는 무효한 매핑을 찾습니다
SELECT 
    b.route_id, 
    b.stop_id
FROM bridge_route_stop b
LEFT JOIN dim_stop s ON b.stop_id = s.stop_id
WHERE s.stop_id IS NULL;

-- 목적: 브릿지 테이블에 연결된 노선 중 유효하지 않은 고아 노선 탐색
-- 2. bridge_route_stop.route_id가 dim_route에 존재하는지 검사 (고아 노선 식별)
-- 브릿지 테이블에는 매핑되어 있으나 실제 노선 마스터 테이블에 존재하지 않는 무효한 매핑을 찾습니다
SELECT 
    b.route_id, 
    b.stop_id
FROM bridge_route_stop b
LEFT JOIN dim_route r ON b.route_id = r.route_id
WHERE r.route_id IS NULL;

-- 목적: 정류장 방문 순서(stop_sequence)의 연속성 및 누락 여부 점검
-- 3. 동일 route_id 안에서 stop_sequence 연속성 검사
-- 정류장 방문 순서(stop_sequence)가 1, 2, 3... 순으로 중간 이빨 빠짐 현상이나 순서 바뀜 없이 이어지는지 찾습니다
WITH seq_check AS (
    SELECT 
        route_id, 
        stop_id,
        stop_sequence,
        LAG(stop_sequence) OVER (PARTITION BY route_id ORDER BY stop_sequence) as prev_seq
    FROM bridge_route_stop
)
SELECT 
    route_id,
    stop_id, 
    stop_sequence, 
    prev_seq
FROM seq_check
WHERE prev_seq IS NOT NULL 
  AND stop_sequence != prev_seq + 1;

-- 목적: 전체적인 노선-정류장 조인 누락 건수 요약 조회
-- 4. route-stop 조인 누락 건수 요약
-- 고립된(어떤 정류장과 매핑되지 않은 노선 / 어떤 노선과도 매핑되지 않은 정류장) 건수를 요약 조회합니다
SELECT 
    (SELECT COUNT(*) 
     FROM dim_route r 
     LEFT JOIN bridge_route_stop b ON r.route_id = b.route_id 
     WHERE b.route_id IS NULL
    ) AS unmapped_routes_count,
    
    (SELECT COUNT(*) 
     FROM dim_stop s 
     LEFT JOIN bridge_route_stop b ON s.stop_id = b.stop_id 
     WHERE b.stop_id IS NULL
    ) AS unmapped_stops_count;
