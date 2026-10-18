"""Pure NumPy evaluation selection and checkpoint retention (no reward changes)."""
import json
from pathlib import Path
import numpy as np


def candidate_rank(result, early=False):
    """Lexicographic: qualification, surviving bilateral progress, survival, tracking.

    This ranks diagnostic candidates; it never changes acceptance thresholds.
    A fallen trial earns no bilateral-progress credit. A dragging/standing trial
    cannot compensate for its missing foot with additional stronger-foot steps.
    """
    m = result['per_trial']
    steps = np.clip(np.min(m['steps'], axis=-1)/3., 0., 1.)
    clearance = np.clip(np.min(m['clear_fraction'], axis=-1)/.08, 0., 1.)
    alternating = np.clip(np.asarray(m['alternation'])/.9, 0., 1.)
    progress = np.mean(np.asarray(m['survived'])*steps*clearance*alternating)
    rank = [float(result['qualified_fraction']), float(progress), float(result['survival_fraction'])]
    if early:
        # Diagnose cold starts even when no trial reaches the five-second quality window.
        rank.append(float(np.mean(m['first_episode_seconds'])))
    if 'incomplete_fraction' in m:
        rank.extend([-float(np.mean(m['incomplete_fraction'])),
                     -float(np.mean(m['repeated_attempt'])),
                     -float(np.mean(m['max_symmetry']))])
    return rank + [-float(result['track_error'])]


def retain_evaluation(agent, result, run):
    """Persist the exact evaluated policy, including baseline and final steps."""
    run = Path(run)
    name = f"eval_{int(result['step'])}.msgpack"
    agent.save(run/name)
    result['checkpoint'] = name
    (run/f"eval_{int(result['step'])}_metrics.json").write_text(json.dumps(result, indent=2)+'\n')
