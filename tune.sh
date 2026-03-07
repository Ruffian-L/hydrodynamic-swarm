#!/bin/bash
# Tuning script for hydrodynamic-swarm physics params
# Runs 10 short generations with varied physics.dt, force_cap, viscosity
# Logs metrics: splats created, avg delta, success tokens, runtime

set -e
BINARY="target/release/hydrodynamic-swarm"
CONFIG="config.toml"
LOG="tuning_results.log"
PROMPT="Explain quantum entanglement simply."
MAX_TOKENS=80
RUNS=10

echo "=== Physics NCA Tuning: $RUNS runs ===" | tee $LOG
echo "Prompt: $PROMPT | Max tokens: $MAX_TOKENS" | tee -a $LOG
echo "Varying: dt, force_cap, viscosity_scale" | tee -a $LOG
echo "" | tee -a $LOG

cp $CONFIG ${CONFIG}.bak

for i in $(seq 1 $RUNS); do
    echo "Run $i/10" | tee -a $LOG
    
    # Vary params slightly (grid-like)
    DT=$(awk "BEGIN {print 0.02 + ($i-1)*0.003}")
    FORCE=$(awk "BEGIN {print 5.0 + ($i*0.5)}")
    VISC=$(awk "BEGIN {print 0.25 + ($i*0.015)}")
    
    # Update config
    sed -i "s/dt = .*/dt = $DT/" $CONFIG
    sed -i "s/force_cap = .*/force_cap = $FORCE/" $CONFIG
    sed -i "s/viscosity_scale = .*/viscosity_scale = $VISC/" $CONFIG
    
    echo "  Params: dt=$DT force=$FORCE visc=$VISC" | tee -a $LOG
    
    # Run with timeout, clear memory for clean test, quiet
    TIMEOUT=30
    START=$(date +%s)
    OUTPUT=$($BINARY --max-tokens $MAX_TOKENS "$PROMPT" --clear-memory 2>&1 | tail -n 50) || true
    END=$(date +%s)
    RUNTIME=$((END-START))
    
    # Extract metrics from output/logs
    SPLATS=$(echo "$OUTPUT" | grep -oE 'splats?[:=]?\s*[0-9]+' | grep -o '[0-9]\+' | tail -1 || echo "0")
    DELTA=$(echo "$OUTPUT" | grep -oE 'delta[_ ]?(norm|mean)[:=]?\s*[0-9.]+' | grep -o '[0-9.]\+' | tail -1 || echo "0")
    TOKENS=$(echo "$OUTPUT" | grep -oE '[0-9]+ tokens' | grep -o '[0-9]\+' | tail -1 || echo "$MAX_TOKENS")
    SUCCESS=$(echo "$OUTPUT" | grep -E 'success|min_success|steer' | wc -l)
    
    echo "  Results: splats=$SPLATS delta=$DELTA tokens=$TOKENS runtime=${RUNTIME}s success_lines=$SUCCESS" | tee -a $LOG
    echo "  Sample log: $(echo "$OUTPUT" | grep -E 'splat|force|delta|steer' | head -2 | tail -1)" | tee -a $LOG
    echo "" | tee -a $LOG
done

mv ${CONFIG}.bak $CONFIG
echo "Tuning complete. Results in $LOG" | tee -a $LOG
echo "Recommended: lower dt (<0.03) for stability, force_cap ~6-8, visc~0.3-0.4" | tee -a $LOG
cat $LOG | tail -20
