"""
project/real_life/model/__init__.py
"""
from .network import RealLifePINN
from .losses import HeteroscedasticLoss, PhysicsInformedLoss

__all__ = ["RealLifePINN", "HeteroscedasticLoss", "PhysicsInformedLoss"]
