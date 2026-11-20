"""Quiet posture features/errors, independent of global XY position and heading."""
import numpy as np


def features(motor_q, rotation, base, soles, xp=np):
    return dict(motor_q=motor_q,
                gravity_body=rotation.T @ xp.asarray([0., 0., -1.]),
                soles_body=(soles-base) @ rotation,
                height_over_soles=base[2]-xp.mean(soles[:, 2]))


def error(current, reference, cfg, xp=np):
    q = current['motor_q']-xp.asarray(reference['motor_q'])
    gravity_dot = xp.sum(current['gravity_body']*xp.asarray(reference['gravity_body']))
    tilt = xp.arccos(xp.clip(gravity_dot, -1., 1.))
    feet = xp.linalg.norm(current['soles_body']-xp.asarray(reference['soles_body']), axis=1)
    height = xp.abs(current['height_over_soles']-reference['height_over_soles'])
    rms = xp.sqrt(xp.mean(q*q))
    maximum = xp.max(xp.abs(q))
    ok = ((rms <= cfg.return_joint_rms_rad) & (maximum <= cfg.return_joint_max_rad)
          & (tilt <= cfg.return_tilt_rad) & (feet.max() <= cfg.return_foot_error_m)
          & (height <= cfg.return_height_error_m))
    score = xp.exp(-xp.mean(q*q)/cfg.return_joint_scale_rad**2
        -tilt**2/cfg.return_tilt_scale_rad**2-xp.mean(feet*feet)/cfg.return_foot_scale_m**2
        -height**2/cfg.return_height_scale_m**2)
    return dict(ok=ok, score=score, joint_rms=rms, joint_max=maximum, tilt=tilt,
                feet_max=feet.max(), height=height)


def initial_phase(xp=np):
    return dict(balance_stable_s=xp.asarray(0., dtype=xp.float32),
                returning=xp.asarray(False), balanced_at=xp.asarray(-1., dtype=xp.float32))
