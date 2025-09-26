"""NI-DAQ digital sync helpers for trial start / code strobes.

Safe to import without NI hardware: gracefully degrades to no-op.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence

try:  # OPTIONAL dependency
    import nidaqmx  # type: ignore
    from nidaqmx.constants import LineGrouping  # type: ignore
except Exception:  # pragma: no cover - absence path
    nidaqmx = None  # type: ignore
    LineGrouping = None  # type: ignore


# ---------------------------- Utility ----------------------------------

def int_to_bits(value: int, n_bits: int) -> List[bool]:
    """Convert integer to list of booleans (LSB first) of length n_bits.

    Raises ValueError if value cannot be represented.
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    max_val = (1 << n_bits) - 1
    if value < 0 or value > max_val:
        raise ValueError(f"Value {value} outside range 0..{max_val} for {n_bits} bits")
    return [bool((value >> i) & 1) for i in range(n_bits)]


# ----------------------------- TrialSync --------------------------------

@dataclass
class TrialSync:
    """Single digital line pulse for marking trial start.

    Example
    -------
    sync = TrialSync(line="Dev1/port0/line0", pulse_ms=2)
    sync.send()  # at trial start
    sync.close()  # on shutdown
    """

    line: str = "Dev1/port0/line0"
    pulse_ms: float = 2.0
    settle_ms: float = 0.1

    def __post_init__(self):
        self._task = None
        if nidaqmx is not None:
            try:
                self._task = nidaqmx.Task()
                self._task.do_channels.add_do_chan(self.line, line_grouping=LineGrouping.CHAN_PER_LINE)
                self._task.write(False)
            except Exception as e:  # pragma: no cover - hardware path
                print(f"[TrialSync] Warning creating task: {e}")
                self._safe_close()
        else:  # pragma: no cover
            print("[TrialSync] nidaqmx not available - running in no-op mode")

    def send(self):
        if self._task is None:
            return
        try:  # pragma: no cover - hardware specific timing
            self._task.write(True)
            time.sleep(self.pulse_ms / 1000.0)
            self._task.write(False)
            if self.settle_ms > 0:
                time.sleep(self.settle_ms / 1000.0)
        except Exception as e:
            print(f"[TrialSync] Pulse failed: {e}")

    def _safe_close(self):  # internal cleanup
        if self._task is not None:
            try:
                self._task.write(False)
            except Exception:
                pass
            try:
                self._task.close()
            except Exception:
                pass
            self._task = None

    def close(self):
        self._safe_close()


# ----------------------------- CodeSync ---------------------------------

class CodeSync:
    """Parallel code output over contiguous digital lines + optional strobe.

    Parameters
    ----------
    data_lines : str
        Contiguous line range e.g. 'Dev1/port0/line0:7' (8 bits).
    strobe_line : str
        Digital line for strobe if `use_separate_strobe=True`.
    use_separate_strobe : bool
        If True, set code on data lines then pulse strobe.
    pulse_ms : float
        Duration of strobe (or code visibility if no separate strobe).
    clear_after : bool
        If True, zero data lines after strobe/hold.
    """

    def __init__(self,
                 data_lines: str = "Dev1/port0/line0:7",
                 strobe_line: str = "Dev1/port0/line7",
                 use_separate_strobe: bool = False,
                 pulse_ms: float = 1.0,
                 clear_after: bool = True):
        self.data_lines = data_lines
        self.strobe_line = strobe_line
        self.use_separate_strobe = use_separate_strobe
        self.pulse_ms = float(pulse_ms)
        self.clear_after = clear_after
        self._task_data = None
        self._task_strobe = None
        if nidaqmx is not None:
            try:
                self._init_tasks()
            except Exception as e:  # pragma: no cover
                print(f"[CodeSync] Warning creating tasks: {e}")
                self.close()
        else:  # pragma: no cover
            print("[CodeSync] nidaqmx not available - running in no-op mode")

    # ------------------ internal helpers ------------------
    def _infer_num_bits(self) -> int:
        if ":" in self.data_lines:
            # pattern ...lineA:B
            try:
                segment = self.data_lines.split("line")[-1]
                start, end = segment.split(":")
                return int(end) - int(start) + 1
            except Exception:
                raise ValueError(f"Cannot parse data_lines '{self.data_lines}'")
        return 1

    def _init_tasks(self):
        self._n_bits = self._infer_num_bits()
        self._task_data = nidaqmx.Task()
        self._task_data.do_channels.add_do_chan(self.data_lines, line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self._task_data.write([False] * self._n_bits)
        if self.use_separate_strobe:
            self._task_strobe = nidaqmx.Task()
            self._task_strobe.do_channels.add_do_chan(self.strobe_line, line_grouping=LineGrouping.CHAN_PER_LINE)
            self._task_strobe.write(False)

    # ------------------ public API ------------------
    @property
    def n_bits(self) -> int:
        return getattr(self, "_n_bits", self._infer_num_bits())

    def send_code(self, code: int):
        if self._task_data is None:  # no-op path
            return
        pattern = int_to_bits(code, self.n_bits)
        try:  # pragma: no cover - hardware interaction
            self._task_data.write(pattern)
            if self.use_separate_strobe and self._task_strobe is not None:
                self._task_strobe.write(True)
                time.sleep(self.pulse_ms / 1000.0)
                self._task_strobe.write(False)
            else:
                time.sleep(self.pulse_ms / 1000.0)
            if self.clear_after:
                self._task_data.write([False] * self.n_bits)
        except Exception as e:
            print(f"[CodeSync] send_code failed: {e}")

    def close(self):
        for t in (self._task_strobe, self._task_data):
            if t is not None:
                try:
                    t.close()
                except Exception:
                    pass
        self._task_data = None
        self._task_strobe = None


__all__ = [
    "TrialSync",
    "CodeSync",
    "int_to_bits",
]


if __name__ == "__main__":  # pragma: no cover
    # Simple test / demo
    trial_start_line = "Dev1/port0/line0"
    stim_onset_line = "Dev1/port0/line1"
    saccade_onset_line = "Dev1/port0/line2"

    sync_pulse_ms = 2.0

    _trial_sync = TrialSync(line=trial_start_line, pulse_ms=sync_pulse_ms)
    _stim_sync = TrialSync(line=stim_onset_line, pulse_ms=sync_pulse_ms)
    _saccade_sync = TrialSync(line=saccade_onset_line, pulse_ms=sync_pulse_ms)

    for i in range(10):
        print(f"Trial {i}")
        _trial_sync.send()
        time.sleep(0.5)
        print("  Stimulus onset")
        _stim_sync.send()
        time.sleep(0.5)
        print("  Saccade onset")
        _saccade_sync.send()
        time.sleep(1.0)
