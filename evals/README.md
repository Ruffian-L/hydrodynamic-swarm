# Named hydro evals

One experiment = one directory. Gemma is a collaborator who scores the **work** hard.

```
./scripts/hydro_eval.sh check climb-after-fail
./scripts/hydro_eval.sh run climb-after-fail
```

`check` is no-GPU and must pass. `run` is Grok's job. Jason is not CI.

Chat order: brief → task → her **insights**, then SCORE/VERDICT as a hard call on the work → peer debrief → she updates SCORE/VERDICT.

SCORE and VERDICT stay. FAIL is allowed as a call on the seat. Do not stamp it on her.

Never `PROMPTS_FILE=<(...)`.
