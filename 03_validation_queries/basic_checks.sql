-- [기본 데이터 무결성 검증]

-- 목적: dim_stop 테이블의 PK(정류장 ID) 중복 여부 확인
-- 1. dim_stop의 stop_id 중복 검사
-- PK 역할을 해야 하는 정류장 ID가 중복으로 적재되었는지 확인합니다 (정상 시 결과 리턴 없음)
SELECT 
    stop_id, 
    COUNT(*) as duplicate_count
FROM dim_stop
GROUP BY stop_id
HAVING COUNT(*) > 1;

-- 목적: dim_route 테이블의 PK(노선 ID) 중복 여부 확인
-- 2. dim_route의 route_id 중복 검사
-- PK 역할을 해야 하는 노선 ID가 중복으로 적재되었는지 확인합니다 (정상 시 결과 리턴 없음)
SELECT 
    route_id, 
    COUNT(*) as duplicate_count
FROM dim_route
GROUP BY route_id
HAVING COUNT(*) > 1;

-- 목적: dim_stop 테이블 내 공간 분석 필수 좌표 누락 검사
-- 3. 좌표(latitude, longitude) NULL 검사
-- 공간 분석에 필수적인 정류장 좌표가 누락된 데이터가 있는지 확인합니다
SELECT 
    stop_id,
    stop_name,
    latitude, 
    longitude
FROM dim_stop
WHERE latitude IS NULL 
   OR longitude IS NULL;

-- 목적: fact_stop_usage_hourly의 물리적 비정상 측정값(음수) 발생 확인
-- 4. fact_stop_usage_hourly의 음수 값 검사
-- 승/하차 인원 등 물리적으로 음수가 될 수 없는 측정값이 발견되는지 확인합니다
SELECT 
    service_date,
    time_bucket,
    stop_id,
    board_count,
    alight_count,
    transfer_count
FROM fact_stop_usage_hourly
WHERE board_count < 0
   OR alight_count < 0
   OR transfer_count < 0;
