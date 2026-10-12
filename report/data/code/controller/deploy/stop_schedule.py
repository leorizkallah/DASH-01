"""STOP-02 command adapter v1, shared by NumPy deployment and JAX training.

The operator requests STOP; the actor continues stepping during BRAKE and receives
its STOP bit only in HOLD. No simulator-only signals are used here.
"""
import numpy as np

VERSION = 1


def duration(v0, decel=0.5, minimum=2.0, xp=np):
    return xp.maximum(minimum, 1.5 * xp.maximum(v0, 0.0) / decel)


def reference(v0, elapsed, seconds, xp=np):
    u = xp.clip(elapsed / xp.maximum(seconds, 1e-6), 0.0, 1.0)
    return v0 * (1.0 - 3.0*u*u + 2.0*u*u*u)


def advance(request, previous, command, estimate, filtered, v0, started, seconds,
            now, dt=0.01, decel=0.5, minimum=2.0, vmax=2.5, xp=np):
    """Pure tick update. Reset with previous=request, seconds=0 for initial HOLD."""
    filtered = filtered + (1.0 - xp.exp(-dt / 0.1)) * (estimate - filtered)
    edge = xp.logical_and(request, xp.logical_not(previous))
    v0 = xp.where(edge, xp.clip(filtered, 0.0, vmax), v0)
    seconds = xp.where(edge, duration(v0, decel, minimum, xp), seconds)
    started = xp.where(edge, now, started)
    elapsed = xp.maximum(0.0, now - started)
    hold = request & (elapsed >= seconds)
    target = xp.where(request, reference(v0, elapsed, seconds, xp), command)
    phase = xp.where(request, xp.where(hold, 2, 1), 0)
    return filtered, v0, started, seconds, target, hold, phase
