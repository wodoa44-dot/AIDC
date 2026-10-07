# 제어 모델 실험

`aidc_control_fair_budget_20260930/`에는 반응형·예측형 제어와 공정 예산 비교가 있다. `control.py`는 controller 구현, `nested/`는 inner tuning 및 outer 평가다.

`aidc_forecast_diagnosis_20261001/`은 예측·제어 통합 진단이다. `core.py`는 공통 물리 executor와 endpoint controller, `mpc.py`는 궤적 MPC, `diagnosis.py`는 입력 교체 및 weight 실험, `review_controller.py`는 기존 결과 후속 분석이다. `outputs/full/`과 `outputs/controller_review_v1/`에 보고서가 있다.

`legacy_e1/`은 metadata 예측 프로젝트에 포함된 초기 제어 코드를 연결한다. 파일을 단독 실행하려면 원래 프로젝트의 환경과 실행 방법을 따른다.

모든 실험 항목은 원본에 대한 심볼릭 링크이며 실행·재개는 원본 README의 경로를 따른다.
