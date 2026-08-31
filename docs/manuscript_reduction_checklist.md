# 원고 감축 체크리스트

현재 36페이지(그림 33개, 표 23개). IEEE Access 통상 분량의 두 배.
아래 항목을 모두 적용하면 **그림 33→30, 표 23→20, 약 3.5~5페이지 감축**
(빈 섹션 집필분 +2.5~3p를 더해도 최종 34~36p).

리뷰어 대응분(§3)은 감축 대상에서 제외했다 — 할당기 포화(V-A, Fig. 11/12)와
yaw 마찰(Fig. 10, 식 (71)~(74))은 지적 사항에 대한 직접 답변이므로 유지한다.

감축 대상은 전부 II·III·IV·V·부록에 있고, 남은 집필은 VIII-C·IX에
있으므로 **순서 의존성이 없다** — 집필 전에 먼저 잘라도 무방하다.

---

## 0. 미완성 잔재 — 내용과 무관, 제출 전 필수

- [ ] 초록 첫 줄 `[TODO: Rev2-C8: Rephrase the inclusion-criteria...]` 삭제
- [ ] 결론 `[TODO: Rev2-C8: Update this improvement summary...]` 삭제
- [ ] 빈 섹션 채우기: **VIII-C-3**(캘리브레이션 민감도),
      **VIII-C-4**(자율비행 과도응답), **IX. LIMITATION** — 현재 제목만 존재
- [ ] 깨진 참조 `??` 정리 (결론에만 6개 이상)
- [ ] 캡션 플레이스홀더: **Fig. 31, Fig. 32, Table 23** 이 "Caption" 상태
- [ ] 초록의 폐기된 선별 기준 서술 교체
      (`ROLL: ϕpeak ≥ 5.0 deg and dx,peak ≥ 0.10 m` / `41.9~70.8%` 등)
      → 전 시행 취합 수치로:
      HGDO 58/57/60/50 %, L1 45/50/42/56 % (ϑ/ω/p/v)
- [ ] 결론 `Simulation studies in Cases E1–E3` → 시뮬은 S 케이스 (표기 오류)
- [ ] 결론의 pivot-based/pivot-free 취합 서술 → pivot-free 단독으로 정정
      (자율비행 결과에서 pivot-based 변형 제외하기로 결정)

### 리뷰어 지적 대응 — 철회할 서술

- [ ] **"z_th = 0.010 m를 채터링 방지를 위해 설정했다"는 서술 삭제.**
      이륙 스위칭은 추력–하중 비교(f_act ≥ W_nom)와 was_airborne 래치만으로
      결정되며 z_th는 이륙 분기에 들어가지 않는다. z_th는 **착륙 판정의
      모호성** 때문에 둔 근접지면 게이트다. 원고의 채터링 방지 주장은
      집필 중 착오이므로 **민감도 분석을 붙여 방어하지 말고 서술을 정정**할 것.

      리뷰어 지적(“hysteresis threshold design is chosen without parameter
      sensitivity analysis to verify its anti-chattering performance”)은
      이 잘못된 문장이 유발한 것이므로, 정정 자체가 답변이 된다.

      뒷받침 사실(원고에 한 문장으로): 알고리듬 4를 그대로 구현해 전 60회
      이륙에 적용하면 `in_flight` 플래그가 **정확히 1회만** 전이하며
      재토글이 없다. 래치 후 추력이 임계 아래로 돌아오지 않기 때문이다
      (통과 후 최대 추력 1.08–1.21 W_nom).

      `docs/tab_switching_sensitivity.tex` 는 **원고에 넣지 않는다** —
      리뷰어가 임계값 선택 근거를 재차 요구할 때만 예비로 사용
      (0.90–1.05 W_nom에서 60/60 무결, 1.10에서 12런 미작동이며 이는 실측
      피크 추력 하한 1.077과 일치).

---

## 1. 감축 대상 (효과 큰 순)

### (a) 피치축 유도 중복 — 약 1컬럼, 그림 1개
식 **(11)~(15)** 는 (4)~(10)의 부호만 바뀐 복제이며, 본문도 이미
"The same argument used for the roll axis applies to the pitch axis"라고
명시하고 있다.
- [ ] (11)(12) 유도 삭제, **결과식 (13)(14)(15b)만** 유지
- [ ] **Fig. 4**(static model along pitch axis) 삭제 — Fig. 2와 대칭

### (b) Section III 표준 재현 압축 — 2~3페이지
(43)~(50) PD-NMPC, (51) 할당, (56)~(63) L1, (64)~(69) HGDO 는 전부
[23], [25]–[28], [32], [33]의 재현.
- [ ] 각 모듈을 핵심 식 2~3개 + 인용으로 축약
- [ ] 고유 기여인 **tanh signum (45)** 는 유지

### (c), (d) — 취소됨: 리뷰어 대응분이므로 유지
Section V-A + Fig. 11/12(할당기 포화)와 Fig. 10 + yaw 마찰 (71)~(74)는
리뷰어 지적에 대한 직접 답변이므로 **삭제하지 말 것**. §3 참조.

### (e) Table 15 (제외 런 30개 개별 나열) — 반 페이지
- [ ] 표 삭제 → "308런 중 30런 게이트 탈락(주로 |ε| 및 linRMSE 초과);
      샘플 게이트는 한 번도 구속되지 않음(Nfull = 64–755)" 한 문장

### (f) Algorithm 2, 3 삭제
- [ ] **Alg. 2** 는 식 (23)과 완전 중복 → 삭제
- [ ] **Alg. 3** 는 2줄짜리 → 문장 하나로 대체
- [ ] **Alg. 1, Alg. 4 는 유지**

### (g) VIII-B4 + Table 17 (로터 동역학 ablation)
변화량 0.03~0.06 mm.
- [ ] **Table 17** 삭제 → 한 문장

### (h) Table 13 (2행짜리 바운드 검증표)
- [ ] 삭제 → 본문 인라인 ("시뮬 252/252, 하드웨어 140/140, 최악비 0.79/0.55")

### (i) Fig. 19 (통신 배선도)
- [ ] 삭제 — 기여와 무관한 구현 세부

### (j) Fig. 16 ↔ Table 7 중복
동일한 케이스 목록·식별가능 영역을 그림과 표로 두 번 제시.
- [ ] 둘 중 하나만 유지 (그림 권장 — 식별 영역이 시각적으로 유효)

### (k) Appendix B 단조성 증명 (B14)~(B17)
본문에서 쓰이는 것은 상한 Rφ < 1/7, RGE < 1/5 뿐.
- [ ] 증명 압축 또는 결과만 인용

### (l) pair-averaging 설명 4회 반복
II-B1, IV-A, V-G, 결론에 같은 논지가 반복.
- [ ] **V-G 한 곳**에만 두고 나머지는 참조로 처리

---

## 2. 추가 감축 후보 (위로 부족할 때만)

- [ ] 자율비행 베어링 그림(`exp_ff_bearing.png`) 제외 → 문장으로 대체
      ("10개 케이스–관측기 쌍 전부에서 비보상 드리프트 베어링이
      로드셀 오프셋 베어링과 25° 이내로 일치, 중앙값 9.5°").
      **조건**: 시뮬 나침반 그림(Fig. 22)은 반드시 유지
- [ ] Section VI-A(로터 모델 + Fig. 14 + Table 5)를 부록으로 이전
      — 시뮬 충실도 근거이므로 본문 유지가 더 안전

---

## 3. 건드리지 말 것

Table 1(관련연구 비교), **Alg. 1**, **Alg. 4**, 식 (7)/(14)/(18),
게이트 (19), 스코어 (23), V-E·V-F·V-G의 바운드 3종,
Table 16/19/20/21/22, Fig. 20/21/24/26/27/28, fc 각주.

### 리뷰어 대응분 — 특히 삭제 금지

이 두 블록은 리뷰어 지적에 대한 직접 답변이므로 축약도 신중히 할 것.

- **Section V-A + Fig. 11, 12** (할당기 포화)
  > "actual rotor thrust allocation has saturation constraints, which
  > distort the linear moment ramp profile during hardware testing.
  > The theoretical model does not incorporate actuator saturation
  > limits…"

  → 이론과 실제 여기신호의 괴리를 메우는 부분. 포화가 일어나지
  않는다는 결론 자체가 답변이므로 그림 2장 유지.

- **Fig. 10 + yaw 마찰 (71)~(74)** (스위칭 로직)
  > "The switching logic … contains ambiguous judgment conditions.
  > The yaw-axis compensation zeroing rule lacks rigorous mathematical
  > demonstration, and hysteresis threshold design is chosen without
  > parameter sensitivity analysis to verify its anti-chattering
  > performance."

  → (71)~(74)가 "yaw zeroing의 수학적 논증", Fig. 10이 "yaw 요구가
  roll/pitch 여유를 잠식한다"는 근거. 둘 다 유지.

---

## 4. 권장 작업 순서

1. **§1 기계적 감축 (a)~(l)** — 결과와 무관, 반나절
2. **§0 빈 섹션 3개 집필** (VIII-C-3, VIII-C-4, IX)
3. **§0 나머지** — 초록·결론 수치 갱신, `??` 참조, 캡션 정리
4. 최종 분량 확인 후 §2 판단

예상 경로: 36p → (1) 30p → (2) 33p → 제출.
