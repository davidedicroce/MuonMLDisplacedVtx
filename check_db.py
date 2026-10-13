import optuna
from optuna.trial import TrialState

storage = "sqlite:////shared/wp2p5/sqlite/optuna_segment_edge_mu200.db"
study = optuna.load_study(
    study_name="segment_edge_mu200",
    storage=storage,
)

for t in study.trials:
    if t.state == TrialState.RUNNING:
        print("Marking stale RUNNING trial as FAIL:", t.number)
        study.tell(t.number, state=TrialState.FAIL)

complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
print("COMPLETE with value:", len(complete))
print("Need more:", max(0, 100 - len(complete)))
