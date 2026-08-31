from dataclasses import dataclass

from mashumaro.mixins.json import DataClassJSONMixin

# a space band of at least this fraction of the line distance always survives, otherwise a
# symbol could never be located in a space at all
MIN_SPACE_FRACTION = 0.1


@dataclass
class PitchDetectionParams(DataClassJSONMixin):
    """Boundaries of the on-line/in-space decision.

    Both values are fractions of the distance between two adjacent staff lines, measured
    *upward from the lower line* of the gap:

        [0, toleranceBottom]              -> on the lower staff line
        [toleranceBottom, 1-toleranceTop] -> in the space
        [1-toleranceTop, 1]               -> on the upper staff line

    The defaults reproduce the formerly hardcoded tolerance of 0.4 half staff spaces exactly.
    """
    toleranceTop: float = 0.3
    toleranceBottom: float = 0.3
    forceClefsOnLine: bool = True

    def clamped(self) -> 'PitchDetectionParams':
        try:
            top = min(max(float(self.toleranceTop), 0.0), 1.0)
            bottom = min(max(float(self.toleranceBottom), 0.0), 1.0)
        except (TypeError, ValueError):
            return PitchDetectionParams()

        if top + bottom > 1 - MIN_SPACE_FRACTION:
            # keep the ratio the user asked for, but leave room for the space
            scale = (1 - MIN_SPACE_FRACTION) / (top + bottom)
            top, bottom = top * scale, bottom * scale

        return PitchDetectionParams(top, bottom, bool(self.forceClefsOnLine))


DEFAULT_PITCH_DETECTION_PARAMS = PitchDetectionParams()
