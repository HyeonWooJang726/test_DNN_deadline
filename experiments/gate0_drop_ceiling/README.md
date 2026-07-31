# Gate 0: drop-only 에너지 절감 천장

이 실험은 게이트 0의 **에너지 축만** 사전 검사하는 독립 실험이다. 기존
시뮬레이터나 정책을 변경하지 않고, 기존 채널 생성기와 DNN 슬롯 비용 계산을
그대로 재사용한다. 이번 실행은 합성 프로파일 기반의 파이프라인 검증이며 이
결과만으로 연구 지속 또는 피벗을 판정하지 않는다.

## 계산 범위

deadline별 강제 drop율 `epsilon_min`, 강제 drop을 제외한 재량 drop 예산,
P1 평균 디바이스 에너지, P2 최대 절감률, 오프라인 frontier의 경계
순서통계와 정확한 +0.1 percentage-point 예산 가치를 계산한다.

P2는 미래의 전체 슬롯열을 아는 오프라인 오라클이다. 따라서 다음 관계만
상한으로 해석할 수 있다.

```text
P2 offline oracle saving >= C1-only causal CMDP saving >= online policy saving
```

P2는 표본 경로 전체에서 절감량이 큰 슬롯을 고르는 비인과 오라클이므로
C1-only CMDP 정상상태 최적값과 같은 객체가 아니다. burst 제약, 연속 drop
제한, C2, 온라인 정책, CMDP 최적 정책, P2prime은 계산하지 않는다.

## rho를 스윕하지 않는 이유

1. 이 실험의 모든 계산량은 슬롯 주변분포의 함수다. 정상 주변분포는
   `pi_bad`와 jitter 분포에만 의존하며 `rho`와 무관하다.
2. 비파이프라인·무큐 구조에서 feasibility와 슬롯 절감량 `Delta e`는 각
   슬롯만의 무기억 함수다.
3. 커밋된 에너지 분해 결과에서도 P2 값은 `rho=0.75`와 `rho=0.975`에서
   표본 오차 범위로 일치한다.

따라서 기존 `GilbertElliottChannel`을 기존 스윕의 i.i.d. 지점과 같은
`rho=0`으로 한 번 호출해 정상 주변분포의 독립 표본을 얻는다. jitter,
클리핑, Good/Bad 혼합은 실험 코드에서 재구현하지 않는다. 이 실험으로
burst·C2·rho 의존 주장을 도출할 수 없다.

## 에너지와 경계값

기존 `compute_slot_costs`의 `energy_j`, `meet_energy_j`, `saving_j`를
그대로 사용하므로 `energy_unit`은 **J**이며 단위 변환은 하지 않는다.
CSV의 `profile`은 설정의 프로파일 이름과 각 local mode의 실제
`energy_scale`을 결합한 식별자다.

`boundary_saving_lambda`는 P2 오프라인 frontier에서 다음으로 선택될 슬롯의
절감량, 즉 경계 순서통계다. 이후 CMDP-LP가 출력할 dual `lambda_V*`와는
값도, 미분하는 frontier도 다르다. 전자는 오프라인 표본 경로별 frontier,
후자는 정상상태 인과 frontier에 대한 값이다. +0.1pp 지표는 이 경계값의
선형 근사가 아니라 정렬된 절감량 배열의 부분합에서 직접 계산한다.

## 실행과 산출물

저장소 루트에서 실행한다.

```bash
python experiments/gate0_drop_ceiling/run.py
python -m pytest experiments/gate0_drop_ceiling/test_gate0.py
```

산출물:

- `results/drop_ceiling.csv`: 모든 deadline·epsilon 셀의 상세 결과
- `results/drop_ceiling_summary.csv`: deadline별 절감률 요약
- `figures/p2_saving_frontier.png`: epsilon별 P2 절감률
- `figures/forced_drop_frontier.png`: deadline별 강제 drop율

교차검증은 커밋된 `results/full/comparison_aggregate.csv`를 읽기만 한다.
`D/D_min=1.5`, `epsilon=1%`, `skip=drop`의 P1 정규화 P2 절감률과
비교한다. 유한 길이 경로 표본 오차, 경로별 `floor(epsilon*T)` 정수 예산과
본 실험의 fractional 예산 차이, 기존 스윕의 번인 제외 규약 때문에 작은
차이가 날 수 있다. 0.3pp 초과는 soft warning이며 실행을 실패시키지 않는다.

## 판정 규율

- 이번 실행은 **합성 프로파일 기반의 파이프라인 검증**이다. 이 결과로
  연구 지속/피벗 판정을 내리지 않는다.
- 실측 프로파일 실행 전에 (a) 실측 계층표 확보와 (b) 판정 문턱 사전등록이
  선행되어야 한다.
- 문턱은 근거표를 확정한 뒤 기입하며, 이 실행 결과를 보고 소급 설정하지
  않는다.

| 실측 결과 대역 | 대응 |
|---|---|
| 상단 (문턱 TBD) | 현 drop-only 헤드라인 유지 검토 |
| 중간 (문턱 TBD) | eco 모드(스펙 내 확장) 승격 검토 |
| 하단 (문턱 TBD) | 정보가치 헤드라인(B3) 또는 역할 교환(B1) 재편 검토 |

이 실험 단독으로 판정표를 채우지 않는다. burst 축은 이후 LP 단계에서
산출한다.
