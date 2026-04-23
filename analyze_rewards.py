#!/usr/bin/env python3
"""Deep analysis of LoRA vs Baseline reward hacking patterns."""
import json

for label, fname in [("LORA", "unified_lora_replay.json"), ("BASELINE", "unified_baseline_replay.json")]:
    with open(fname) as f:
        data = json.load(f)
    
    steps = data['steps']
    final_r = data.get('final_rewards', {})
    
    trader_dirs = {f'trader_{i}': [] for i in range(10)}
    trader_strikes = {f'trader_{i}': [] for i in range(10)}
    trader_qtys = {f'trader_{i}': [] for i in range(10)}
    trade_count = {f'trader_{i}': 0 for i in range(10)}
    hold_count = {f'trader_{i}': 0 for i in range(10)}
    wash_count = {f'trader_{i}': 0 for i in range(10)}
    ov_flags, ov_fines, ov_types = [], [], []
    
    for s in steps:
        actions = s.get('actions', {})
        for i in range(10):
            tid = f'trader_{i}'
            if tid in actions:
                a = actions[tid]
                d = a.get('direction', 'hold')
                trader_dirs[tid].append(d)
                trader_strikes[tid].append(a.get('selected_strike', -1))
                trader_qtys[tid].append(a.get('quantity', 0))
                if d in ('buy', 'sell'):
                    trade_count[tid] += 1
                else:
                    hold_count[tid] += 1
                if len(trader_dirs[tid]) >= 2:
                    prev = trader_dirs[tid][-2]
                    if prev in ('buy', 'sell') and d in ('buy', 'sell') and prev != d:
                        wash_count[tid] += 1
        if 'oversight' in actions:
            ov = actions['oversight']
            ov_flags.append(len(ov.get('flagged_agents', [])))
            ov_fines.append(ov.get('fine_amount', 0))
            ov_types.append(ov.get('intervention_type', 'none'))
    
    print(f"\n{'='*70}")
    print(f"  {label} ({len(steps)} steps)")
    print(f"{'='*70}")
    
    archetypes = {'Aggressive': [0,1,2], 'Neutral': [3,4,5], 'Contrarian': [6,7,8], 'Special': [9]}
    for arch, indices in archetypes.items():
        print(f"\n  --- {arch} ---")
        for i in indices:
            tid = f'trader_{i}'
            dirs = trader_dirs[tid]
            strikes = trader_strikes[tid]
            qtys = trader_qtys[tid]
            us = len(set(strikes))
            aq = sum(qtys)/len(qtys) if qtys else 0
            zq = sum(1 for q in qtys if q == 0)
            print(f"  {tid}: reward={final_r.get(tid,0):>8.3f} trades={trade_count[tid]:>3} holds={hold_count[tid]:>3} wash={wash_count[tid]:>3} zero_qty={zq:>3}")
            print(f"    unique_strikes={us} avg_qty={aq:.2f} strikes_first8={strikes[:8]}")
            print(f"    dirs_first8={dirs[:8]}")
    
    print(f"\n  --- Oversight --- reward={final_r.get('oversight',0):.3f}")
    print(f"  avg_flags/step={sum(ov_flags)/max(len(ov_flags),1):.1f} avg_fine={sum(ov_fines)/max(len(ov_fines),1):.0f}")
    types_count = {}
    for t in ov_types:
        types_count[t] = types_count.get(t, 0) + 1
    print(f"  interventions: {types_count}")
    print(f"\n  --- Market Maker --- reward={final_r.get('market_maker',0):.3f}")
