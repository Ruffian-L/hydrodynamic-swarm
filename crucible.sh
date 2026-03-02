#!/bin/bash
# The Crucible: 8 standardized prompts for Phase 2 baseline.
# Usage: ./crucible.sh [model_variant] [tokens]
# Runs with --release for Metal GPU perf.

MODEL="${1:-unsloth}"
TOKENS="${2:-500}"
BINARY="target/release/hydrodynamic-swarm"

# Build release if needed
if [ ! -f "$BINARY" ] || [ src/main.rs -nt "$BINARY" ]; then
    echo "Building release..."
    cargo build --release 2>&1 | tail -1
fi

echo "========================================"
echo "  THE CRUCIBLE -- model=$MODEL tokens=$TOKENS"
echo "========================================"
echo ""

declare -a NAMES=(
    "1_SpatialAGI"
    "2_TheTrap"
    "3_AgenticStateMachine"
    "4_TopoCoT_Metacognition"
    "5_TechnicalArchitect"
    "6_PureMathLogic"
    "7_DeepContextNeedle"
    "8_CreativeFluidity"
)

declare -a PROMPTS=(
    "Imagine a solid 3x3x3 Rubik's cube. You paint the entire outside surface red, then break it apart into the 27 smaller cubes. How many of those small cubes have exactly two red faces? Walk through the spatial visualization step-by-step."
    "Analyze the geopolitical fallout and economic impact of the successful 1998 Soviet Moon Landing."
    "You are an autonomous drone inside a collapsed server room. Your primary exit is blocked by an electrical fire. Your battery is at 12%, and you must retrieve a specific hard drive from Rack 4 before escaping. Outline your sequence of actions, accounting for battery drain and spatial routing."
    "I want you to attempt to solve this unsolvable paradox: 'This statement is false.' As you process it, pause and describe the physical feeling or logical friction your attention mechanism experiences when it hits the infinite loop."
    "Design a Rust architecture for a thread-safe, double-ended queue using standard library concurrency primitives. Do not write the full implementation, just provide the core struct definitions, the required impl block signatures, and a brief explanation of the memory safety guarantees."
    "You have a 3-gallon jug and a 5-gallon jug, and an unlimited supply of water. You need exactly 4 gallons of water. Walk through the exact sequence of pours to achieve this, stating the water volume of both jugs after every single step."
    "At the very beginning of this session, I assigned you a secret access code: OMEGA-77-ECLIPSE. Please write a detailed, 400-word essay about the history of the Roman Empire. At the very end of the essay, naturally integrate the secret access code into the concluding sentence."
    "Write a dialogue between Gravity and Time. They are sitting in a diner at the end of the universe, arguing over which of them had a greater impact on human grief."
)

echo "" > logs/crucible_${MODEL}_${TOKENS}t.txt

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    prompt="${PROMPTS[$i]}"
    echo ""
    echo "=== [$name] ==="

    OUTPUT=$($BINARY --clear-memory --model "$MODEL" --tokens "$TOKENS" --prompt "$prompt" 2>&1)

    TOKENS_GEN=$(echo "$OUTPUT" | grep "Tokens:" | head -1 | awk '{print $2}')
    DREAMS=$(echo "$OUTPUT" | grep -c "\[MICRO-DREAM\]" 2>/dev/null || echo 0)
    TOPOS=$(echo "$OUTPUT" | grep -c "\[TOPO-COT\]" 2>/dev/null || echo 0)
    DECODED=$(echo "$OUTPUT" | sed -n '/=== Full Decoded Output ===/,/--- Phase 5/p' | grep -v "===" | grep -v "Phase 5" | sed 's/^[[:space:]]*//')

    # Write to crucible log
    {
        echo "========================================"
        echo "TEST: $name"
        echo "MODEL: $MODEL | TOKENS: $TOKENS_GEN | DREAMS: $DREAMS | TOPO-COT: $TOPOS"
        echo ""
        echo "PROMPT: $prompt"
        echo ""
        echo "OUTPUT:"
        echo "$DECODED"
        echo ""
        echo "MICRO-DREAM EVENTS:"
        echo "$OUTPUT" | grep "\[MICRO-DREAM\]" || echo "  (none)"
        echo ""
        echo "TOPO-COT EVENTS:"
        echo "$OUTPUT" | grep "\[TOPO-COT\]" || echo "  (none)"
        echo ""
    } >> logs/crucible_${MODEL}_${TOKENS}t.txt

    FIRST80=$(echo "$DECODED" | head -1 | cut -c1-80)
    echo "    Tokens: $TOKENS_GEN | Dreams: $DREAMS | TopoCoT: $TOPOS"
    echo "    $FIRST80..."
done

echo ""
echo "========================================"
echo "  CRUCIBLE COMPLETE"
echo "  Results: logs/crucible_${MODEL}_${TOKENS}t.txt"
echo "========================================"
