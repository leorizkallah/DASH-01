"""Host-side recovery promotion; no simulator dependencies."""


def update(agent, result):
    cur, c = agent.cur, agent.cfg
    passed = (result['hard_trials'] >= 200 and result['hard_rate'] >= .90
              and min(result['sector_rates']) >= .80 and result['quiet_rate'] >= .95)
    cur['push_ema'] = result['hard_rate']
    cur['probe_quiet_rate'] = result['quiet_rate']
    cur['probe_worst_sector'] = min(result['sector_rates'])
    cur['probe_streak'] = cur.get('probe_streak', 0)+1 if passed else 0
    poor = result['hard_rate'] < .70 or result['quiet_rate'] < .90
    cur['probe_poor_streak'] = cur.get('probe_poor_streak', 0)+1 if poor else 0
    moved = None
    if cur['probe_poor_streak'] >= 2:
        cur['push_level'] = max(.02, cur['push_level']*.80)
        cur['probe_poor_streak'] = cur['probe_streak'] = 0
        moved = f"recovery retreat -> {cur['push_level']:.4f} m/s (quiet {result['quiet_rate']:.3f})"
    elif cur['probe_streak'] >= 2:
        if cur.get('push_stage', 0) < 3:
            cur['push_stage'] = cur.get('push_stage', 0)+1
            moved = f"disturbance variety stage -> {cur['push_stage']} (same impulse level)"
        else:
            cur['push_level'] = min(c.push_level_max, cur['push_level']*c.push_level_up)
            cur['n_promote'] += 1
            moved = f"recovery impulse -> {cur['push_level']:.4f} m/s"
        cur['probe_streak'] = cur['probe_poor_streak'] = 0
    return moved
