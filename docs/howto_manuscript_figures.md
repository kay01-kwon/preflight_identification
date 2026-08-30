# 하드웨어 원고 그림 재생성 가이드

세 그림 — 임계 모멘트 정적 비교(`fig_mcrit_static.png`), 피팅 비교
(`exp_fit_comparison.png`), CoM 오프셋 추정 오차(`exp_estimator_err.png`) —
의 재생성 절차. 모든 명령은 **저장소 루트에서** 실행한다.

## 0. 공통 준비

의존성: `numpy`, `scipy`, `matplotlib`, `ruptures`(CPD 벤치마크),
그리고 bag 읽기용 ROS 2 파이썬 패키지(`rosbag2_py` 등).
`DataSet/exp/`(하드웨어 캠페인 140런)가 있어야 한다.

그림 스크립트는 중간 CSV 두 개를 읽는다. 임의의 작업 폴더(아래
`scratch/`)에 먼저 만들어 둔다. **PNLS(free fit) 열이 들어가므로,
시드 복원 커밋(`edd9e54`) 이후 코드로 새로 만든 CSV여야 한다** —
그 전에 뽑아 둔 CSV는 PNLS 값이 다르다.

```bash
mkdir -p scratch

# (a) 6개 검출기 벤치마크 → nls_comparison_runs.csv (약 1시간)
python analysis/nls_comparison.py scratch

# (b) GE 예측/잔차 → mcrit_prediction.csv (수 분)
python analysis/mcrit_prediction.py scratch
```

## 1. 임계 모멘트 정적 비교 (2×1, Mx/My × E1–E5)

6개 검출기의 방향별 평균(95% 경험 구간 수염)과 GE 이론선 3종
(no ground effect / Single rotor / Rotor interference)을 겹친 그림.

```bash
python analysis/mcrit_static_figure.py scratch --outdir docs --dpi 600
```

- 입력: `scratch/mcrit_prediction.csv` + `scratch/nls_comparison_runs.csv`
- 출력: `docs/fig_mcrit_static.png` (PNG만)

## 2. 피팅 비교 예시런 (E2/My+, 1.2 N·m/s)

측정 ω 위에 COSH(제약) 피팅, PNLS(자유) 피팅, GE 보정 정적 임계선을
겹친 단일런 그림. 수치는 레전드에 함께 표기된다.

```bash
python analysis/exp_fit_comparison.py scratch --outdir docs --dpi 600
```

- 입력: `scratch/mcrit_prediction.csv` (GE 보정 임계값 M_crit,th 조회용;
  두 피팅은 스크립트가 bag에서 직접 수행)
- 출력: `docs/exp_fit_comparison.png`
- 시드 복원 후 PNLS 온셋은 +1.137 N·m로 나온다
  (구버전 그림의 +0.272는 시드 제거 시절 값).

## 3. CoM 오프셋 추정 오차 (2×1, x_off/y_off × E1–E5)

케이스별 6개 검출기의 성분 오차(평균 추정 − 로드셀 참값)와 Welch-t
95% 신뢰구간. 같은 스크립트가 COSH 개별-대-평균 그림도 함께 만든다.

```bash
python analysis/estimator_error_figure.py scratch --outdir docs --dpi 600
```

- 입력: `scratch/nls_comparison_runs.csv`
- 출력: `docs/exp_estimator_err.png`, `docs/exp_estimator_indiv.png` (PNG만)

## 요약표

| 그림 | 스크립트 | 입력 CSV |
|---|---|---|
| `fig_mcrit_static.png` | `analysis/mcrit_static_figure.py` | `mcrit_prediction.csv`, `nls_comparison_runs.csv` |
| `exp_fit_comparison.png` | `analysis/exp_fit_comparison.py` | `mcrit_prediction.csv` |
| `exp_estimator_err.png`, `exp_estimator_indiv.png` | `analysis/estimator_error_figure.py` | `nls_comparison_runs.csv` |
| (참고) `exp_score_sensitivity.png`, `exp_score_heatmap.png` | `analysis/score_sensitivity.py` (`--outdir docs --dpi 600`) | 없음 — bag에서 직접 |
| (참고) `exp_score_joint.png` | `analysis/score_sensitivity.py --joint` (데이터셋별 병렬, 4코어 ~25분) | 없음 — bag에서 직접 |

모든 그림 스크립트는 `--outdir`(기본 `docs`)와 `--dpi`(기본 600)를 받는다.
표 재생성은 `analysis/cv_case_table.py scratch/nls_comparison_runs.csv`,
`analysis/mcrit_method_diff.py scratch/nls_comparison_runs.csv`,
`analysis/offset_error_table.py scratch` — 모두 `--outdir docs`.
