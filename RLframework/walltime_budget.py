"""Wall-clock stopping and checkpoint reservation, independent of environment steps."""
import time


class WalltimeBudget:
    def __init__(self, hours, deadline_epoch=None, reserve_s=600., clock=time.monotonic,
                 epoch_clock=time.time):
        if hours <= 0 or reserve_s < 0 or reserve_s >= hours*3600:
            raise ValueError('positive hours and a smaller nonnegative reservation required')
        self.clock = clock
        self.started = clock()
        seconds = hours*3600
        if deadline_epoch is not None:
            seconds = min(seconds, deadline_epoch-epoch_clock())
        self.deadline = self.started+max(seconds, 0.)
        self.reserve_s = reserve_s
        self.signal_received = None

    def request_stop(self, signum, frame=None):
        self.signal_received = signum

    def remaining(self):
        return max(0., self.deadline-self.clock())

    def training_allowed(self):
        return self.signal_received is None and self.remaining() > self.reserve_s

    def evaluation_allowed(self):
        return self.signal_received is None and self.remaining() > self.reserve_s+300.

    def summary(self):
        return dict(elapsed_process_s=self.clock()-self.started, remaining_s=self.remaining(),
                    reserve_s=self.reserve_s, signal=self.signal_received,
                    reason='signal' if self.signal_received else 'walltime_budget')
