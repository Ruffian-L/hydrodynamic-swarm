#!/bin/bash
# Ablation sweep script
# Runs each config and extracts key metrics

BASEDIR="/media/ruffianl/ghost_team/projects/hydrodynamic-swarm"
BINARY="$BASEDIR/target/release/hydrodynamic-swarm"
PROMPT="Explain the Physics of Friendship in one paragraph."
RESULTS_FILE="$BASEDIR/logs/ablation_results.tsv"

echo -e "config\tsteps\texit_code\toutput_length\tgoal_norm\tsplat_before\tsplat_after\tdelta_mean" > "$RESULTS_FILE"

run_config() {
    local config_name=$1
    local config_file=$2
    
    echo "=== Running: $config_name ==="
    
    # Run the binary with this config
    output=$($BINARY --config "$config_file" --prompt "$PROMPT" 2>&1)
    exit_code=$?
    
    echo "Exit code: $exit_code"
    
    # Find the latest log file for this run
    log_dir=$(grep "log_dir" "$config_file" | sed 's/.*"\(.*\)".*/\1/')
    latest_jsonl="$log_dir/latest.jsonl"
    
    if [ -f "$latest_jsonl" ]; then
        # Extract metrics from the summary line
        summary=$(grep '"entry_type":"summary"' "$latest_jsonl" | tail -1)
        
        if [ -n "$summary" ]; then
            steps=$(echo "$summary" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('summary',{}).get('generated_token_count',0))" 2>/dev/null)
            output_len=$(echo "$summary" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('summary',{}).get('decoded_output','')))" 2>/dev/null)
            goal_norm=$(echo "$summary" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('summary',{}).get('goal_attractor_norm',0))" 2>/dev/null)
            splat_before=$(echo "$summary" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('summary',{}).get('splat_count_before',0))" 2>/dev/null)
            splat_after=$(echo "$summary" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('summary',{}).get('splat_count_after',0))" 2>/dev/null)
            delta_mean=$(echo "$summary" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('summary',{}).get('delta_mean',0))" 2>/dev/null)
        else
            steps=0; output_len=0; goal_norm=0; splat_before=0; splat_after=0; delta_mean=0
        fi
    else
        steps=0; output_len=0; goal_norm=0; splat_before=0; splat_after=0; delta_mean=0
    fi
    
    echo -e "$config_name\t$steps\t$exit_code\t$output_len\t$goal_norm\t$splat_before\t$splat_after\t$delta_mean" >> "$RESULTS_FILE"
    echo "Results: steps=$steps exit=$exit_code output_len=$output_len goal_norm=$goal_norm splat=$splat_before->$splat_after delta_mean=$delta_mean"
    echo ""
}

# Run baseline first
run_config "baseline" "$BASEDIR/config.toml"

# Run ablation configs
run_config "force3" "$BASEDIR/config_ablation_force3.toml"
run_config "T02" "$BASEDIR/config_ablation_T02.toml"
run_config "highdecay" "$BASEDIR/config_ablation_highdecay.toml"
run_config "lowdecay" "$BASEDIR/config_ablation_lowdecay.toml"
run_config "pain1" "$BASEDIR/config_ablation_pain1.toml"

echo "=== Sweep complete ==="
echo "Results saved to: $RESULTS_FILE"
cat "$RESULTS_FILE"
