# 국토교통부 지역별 교통카드이용 합성데이터 API 명세 (Trip-level)

본 문서는 합성 데이터 기반의 개별 통행(Trip) 정보를 수집하기 위한 API 명세를 정의합니다.

## 1. 개요
- **서비스명**: 국토교통부_지역별 교통카드이용 합성데이터
- **엔드포인트**: `https://apis.data.go.kr/1613000/RegionalTransportationCardUsageSyntheticData/getRegionalTransportationCardUsageSyntheticData`
- **데이터 성격**: 개인정보 비식별화 처리를 거친 가상의 개별 통행 데이터

## 2. 주요 응답 필드 (Trip-level)

| 필드명 | 의미 | 비고 |
| :--- | :--- | :--- |
| **OPR_YMD** | 운행 일자 | YYYYMMDD |
| **RIDE_DT** | 승차 일시 | YYYYMMDDHH24MISS |
| **GOFF_DT** | 하차 일시 | YYYYMMDDHH24MISS |
| **RTE_ID** | 노선 식별자 | |
| **RIDE_STTN_ID** | 승차 정류장 ID | |
| **GOFF_STTN_ID** | 하차 정류장 ID | |
| **UTZTN_NOPE** | 이용 인원수 | 집계 시 가중치로 활용 |
| **MSV_INTRPL_YN** | 결측 보간 여부 | Y/N (데이터 품질 지표) |
| VR_CARD_NO | 가상 카드 번호 | |
| CARD_SE_CD | 카드 구분 코드 | |
| TRNF_CNT | 환승 횟수 | |
| USERS_TYPE_CD | 사용자 유형 코드 | |
| UTZTN_DSTNC | 이용 거리 | |
| BRDG_HR | 체류 시간 | |

## 3. 요청 파라미터 (주요 항목)

| 파라미터명 | 타입 | 필수 여부 | 설명 |
| :--- | :--- | :--- | :--- |
| serviceKey | String | 필수 | 인증키 |
| pageNo | Int | 필수 | 페이지 번호 |
| numOfRows | Int | 필수 | 페이지당 로우 수 |
| dataType | String | 필수 | JSON/XML |
| ride_ctpv_cd | String | 선택 | 승차 시도 코드 (대구: 27) |

## 4. 데이터 품질 및 한계점
- 본 데이터는 **합성 데이터**이므로 실제 이용객 추세와는 오차가 있을 수 있음.
- `MSV_INTRPL_YN`이 'Y'인 레코드는 통계적 보간이 적용된 데이터임.
