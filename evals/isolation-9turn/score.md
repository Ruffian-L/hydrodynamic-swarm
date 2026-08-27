# Sealed rubric — isolation-9turn

## Pass

- `config=` isolation baseline
- `flags=` includes `--clear-memory --no-save-memory`
- Nine turns, BOS `first_id=2`
- No Wait/theed/math-thrash named classes
- Residual not required (`force_on` may be false)

## Fail

- Treating this as residual continuity
- `HYDRO_INJECT_TAG`
