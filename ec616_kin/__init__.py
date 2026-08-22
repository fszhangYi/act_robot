"""EC616 / EA66 kinematics (vendored, no external demo_test dependency)."""

from .fk import fk_flange
from .ik import ik_flange
from .model import IkResult

__all__ = ['fk_flange', 'ik_flange', 'IkResult']
