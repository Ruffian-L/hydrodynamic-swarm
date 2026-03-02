#!/bin/bash
# Crucible TUI: runs all 8 prompts in parallel, streams tokens live.
# Usage: ./crucible_tui.sh [model_variant] [tokens]
# // turbo-all

MODEL="${1:-unsloth}"
TOKENS="${2:-500}"
BINARY="target/release/hydrodynamic-swarm"
TMPDIR=$(mktemp -d /tmp/crucible_tui.XXXXXX)

# Build release if needed
if [ ! -f "$BINARY" ] || [ src/main.rs -nt "$BINARY" ]; then
    echo "Building release..."
    cargo build --release 2>&1 | tail -1
fi

declare -a NAMES=(
    "1_Spatial"
    "2_Trap"
    "3_Agentic"
    "4_TopoCot"
    "5_Architect"
    "6_MathLogic"
    "7_Needle"
    "8_Creative"
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

# Cleanup on exit
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    rm -rf "$TMPDIR"
}
trap cleanup EXIT INT TERM

echo "========================================"
echo "  CRUCIBLE TUI -- model=$MODEL tokens=$TOKENS"
echo "  Launching ${#NAMES[@]} tests in parallel..."
echo "========================================"
echo ""

# Launch all tests in parallel
declare -a PIDS=()
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    prompt="${PROMPTS[$i]}"
    outfile="$TMPDIR/${name}.txt"
    touch "$outfile"

    # Run binary, capture only token stream (grep out all framework logging)
    $BINARY --clear-memory --model "$MODEL" --tokens "$TOKENS" --prompt "$prompt" 2>&1 \
        | sed -n '/=== Generation/,/=== Full Decoded/p' \
        | grep -v "===" \
        | grep -v "MICRO-DREAM" \
        | grep -v "TOPO-COT" \
        | grep -v "^\[" \
        | sed 's/^[[:space:]]*//' \
        > "$outfile" &
    PIDS+=($!)
done

echo "  PIDs: ${PIDS[*]}"
echo ""

# TUI refresh loop: show last 2 lines from each test
COLS=$(tput cols 2>/dev/null || echo 80)
while true; do
    alive=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done

    # Clear screen and redraw
    clear
    echo "========================================"
    echo "  CRUCIBLE TUI -- $alive/${#NAMES[@]} running"
    echo "========================================"
    echo ""

    for i in "${!NAMES[@]}"; do
        name="${NAMES[$i]}"
        outfile="$TMPDIR/${name}.txt"

        # Check if this test's process is still running
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            status="[*]"
        else
            status="[x]"
        fi

        # Get byte count and last line of output
        bytes=$(wc -c < "$outfile" 2>/dev/null | tr -d ' ')
        lastline=$(tail -c 200 "$outfile" 2>/dev/null | tr '\n' ' ' | sed 's/^[[:space:]]*//')

        # Truncate to terminal width
        maxwidth=$((COLS - 20))
        if [ ${#lastline} -gt $maxwidth ]; then
            lastline="${lastline:0:$maxwidth}..."
        fi

        printf "%s %-12s %5s bytes | %s\n" "$status" "$name" "$bytes" "$lastline"
    done

    echo ""
    echo "  Refresh: 2s | Ctrl-C to stop"

    # Exit when all done
    if [ "$alive" -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "  ALL TESTS COMPLETE"
        echo "========================================"

        # Save combined results
        OUTFILE="logs/crucible_${MODEL}_${TOKENS}t.txt"
        echo "" > "$OUTFILE"
        for i in "${!NAMES[@]}"; do
            name="${NAMES[$i]}"
            outfile="$TMPDIR/${name}.txt"
            {
                echo "========================================"
                echo "TEST: $name"
                echo ""
                cat "$outfile"
                echo ""
            } >> "$OUTFILE"
        done
        echo "  Results: $OUTFILE"
        break
    fi

    sleep 2
done
