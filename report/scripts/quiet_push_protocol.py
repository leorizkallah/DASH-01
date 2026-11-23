"""Pure push-screen protocol; newton-valued cases and exact control-tick timing."""
import numpy as np

FORCES=(0.,2.,5.,10.,15.,20.,30.,40.,60.)
DIRECTIONS=(('forward',0.),('backward',np.pi),('left',np.pi/2),('right',-np.pi/2))
DT=.01
PUSH_TICK=600
PULSE_TICKS=10
END_TICK=2000
HOLD_TICKS=500


def cases():
    result=[dict(force_n=0.,direction='control',azimuth_rad=0.,repeat=r) for r in range(2)]
    for force in FORCES[1:]:
        for direction,azimuth in DIRECTIONS:
            for r in range(2):result.append(dict(force_n=force,direction=direction,azimuth_rad=azimuth,repeat=r))
    return [dict(index=i,**c) for i,c in enumerate(result)]


def force_vector(rotation,magnitude,azimuth,xp=np):
    forward=rotation[:,0]*xp.asarray([1.,1.,0.])
    forward=forward/xp.maximum(xp.linalg.norm(forward),1e-8)
    left=xp.stack([-forward[1],forward[0],xp.asarray(0.)])
    return magnitude*(xp.cos(azimuth)*forward+xp.sin(azimuth)*left)


def active(tick):return (tick>=PUSH_TICK)&(tick<PUSH_TICK+PULSE_TICKS)


def quiet_counts(old,loaded_old,quiet,loaded,tick,xp=np):
    # First quiet interval is entirely after the pulse, never carried from warm-up.
    permitted=tick>=PUSH_TICK+PULSE_TICKS
    streak=xp.where(permitted&quiet,old+1,0)
    loading=xp.where(permitted&quiet,loaded_old+loaded,0)
    ok=(streak>=HOLD_TICKS)&(20*loading>=19*streak)
    return streak,loading,ok
