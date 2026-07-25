# A/B vanilla vs hydro

- model: `/media/ruffianl/ghost_team/projects/hydrodynamic-swarm/data/google/gemma-3-4b-it-Q4_K_M.gguf`
- tokens: 40 · temp: 0.88
- vanilla: llama-server HTTP :8211 (not llama-cli)
- prompt: Explain the Physics of Friendship in one short paragraph.
- B exit: 0

## A — vanilla (no physics)
```
The physics of friendship can be surprisingly analogous to physical systems – it’s largely governed by attraction and mutual reinforcement. Like gravitational forces, shared interests, values, and emotional connection create an initial attraction,
```

## B — hydro surface
```
Friendship is a social relationship based on physics, friendship can be defined as an intricate and bonding between two individuals's are characterized by mutual attraction to shared emotional dependent upon physic relationships that’s
```

## Read
- A≈B wording → mostly **base model**.
- B different spine / re-anchor after pain → **pull**.
