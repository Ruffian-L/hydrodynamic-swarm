# Us run card (stateless)

“I” = us. Logs and scars remember; we don’t perform claim theater.

## Two mouths, one hiker

| Body | Role |
|------|------|
| **Speaker** (`hydrodynamic-swarm` + GGUF) | Residual walk / native geometry / hiker |
| **Enzyme** (`TextEnzyme` / `scripts/endocrine_enzyme.sh`) | Short cold text only → blooms (not “FunctionGemma” costume) |
| **TermSplat** | Paint FieldFrames only |

**One shot (speaker + weather):** `./scripts/us_dual.sh` · optional `START_ENZYME=1 ./scripts/us_dual.sh`

## Terminal A — enzyme (optional)

```bash
cd hydrodynamic-swarm
./scripts/endocrine_enzyme.sh start
./scripts/endocrine_enzyme.sh status
export ENDOCRINE_URL=http://127.0.0.1:8210/v1
export ENDOCRINE_MODEL=local
```

## Terminal B — speaker

```bash
cd hydrodynamic-swarm
# same shell needs ENDOCRINE_URL if you want [ENZYME] not [FACT #]
./run_swarm.sh "Explain the Physics of Friendship in one short paragraph." 40
```

Look for: `native=true` · `[BLOOM native]` · optional `[ENZYME]` · `TermSplat weather: logs/…`

## Terminal C — weather

```bash
cd termsplat
./target/release/termsplat pipe ../hydrodynamic-swarm/logs/latest.termsplat.jsonl --follow
```

## Stop enzyme

```bash
./scripts/endocrine_enzyme.sh stop
```
