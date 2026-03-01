#!/bin/bash
# 5-Prompt Sweep: runs all 5 categories on a given model variant.
# Usage: ./sweep.sh [model_variant]
# Default model: unsloth

MODEL="${1:-unsloth}"
echo "=== 5-PROMPT SWEEP: model=$MODEL ==="
echo ""

declare -a CATEGORIES=("Physics" "Technical" "Creative" "Reasoning" "Abstract")
declare -a PROMPTS=(
    "Explain the Physics of Friendship in one paragraph."
    "How does transformer attention work in detail?"
    "Write a short poem about gravity and human connection."
    "Why do birds fly south for winter? Give a multi-step explanation."
    "What is consciousness?"
)

declare -a R_CAT=()
declare -a R_TOK=()
declare -a R_TXT=()

for i in "${!CATEGORIES[@]}"; do
    cat="${CATEGORIES[$i]}"
    prompt="${PROMPTS[$i]}"
    echo "--- [$cat] ---"
    echo "    Prompt: \"$prompt\""

    OUTPUT=$(cargo run -- --clear-memory --model "$MODEL" --prompt "$prompt" 2>&1)
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "    ERROR: cargo run failed (exit $STATUS)"
        R_CAT+=("$cat")
        R_TOK+=("ERR")
        R_TXT+=("FAILED")
        continue
    fi

    TOKENS=$(echo "$OUTPUT" | grep "Tokens:" | head -1 | awk '{print $2}')
    DECODED=$(echo "$OUTPUT" | sed -n '/=== Full Decoded Output ===/,/--- Phase 5/p' | grep -v "===" | grep -v "Phase 5" | head -3 | tr '\n' ' ' | sed 's/^[[:space:]]*//' | cut -c1-80)

    echo "    Tokens: $TOKENS"
    echo "    Output: ${DECODED}..."
    echo ""

    R_CAT+=("$cat")
    R_TOK+=("$TOKENS")
    R_TXT+=("$DECODED")
done

echo ""
echo "=== SWEEP RESULTS: $MODEL ==="
echo ""
printf "%-12s | %-6s | %s\n" "Category" "Model" "First 80 chars"
printf "%-12s-+-%-6s-+-%s\n" "------------" "------" "----------------------------------------"
for i in "${!R_CAT[@]}"; do
    printf "%-12s | %-6s | %s\n" "${R_CAT[$i]}" "$MODEL" "${R_TXT[$i]}"
done
echo ""
echo "=== SWEEP COMPLETE ==="
