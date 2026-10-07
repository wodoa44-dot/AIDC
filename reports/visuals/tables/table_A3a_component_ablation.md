**Table A3a_component_ablation. Component comparison at 200 ms sampling — run-macro CRPS (W).**

| Model         |   H=200 ms |   H=1000 ms |   H=2000 ms |
|:--------------|-----------:|------------:|------------:|
| Unconditional |     755.28 |      751.88 |      748.68 |
| Current only  |     129.15 |      371.54 |      435.81 |
| History       |     132.82 |      183.6  |      195.41 |

_행 간에는 입력 구성만 달라진다. Unconditional은 입력 없음, Current only는 최신 bin, History는 선택된 causal history feature를 사용한다._
