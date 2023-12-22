
from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class Z1View(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "Z1View",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )
        
        self._link00 = RigidPrimView(prim_paths_expr="/World/envs/.*/z1/link00", name="link00_view", reset_xform_properties=False)
        self._link01 = RigidPrimView(prim_paths_expr="/World/envs/.*/z1/link01", name="link01_view", reset_xform_properties=False)
        self._link02 = RigidPrimView(prim_paths_expr="/World/envs/.*/z1/link02", name="link02_view", reset_xform_properties=False)
        self._link03 = RigidPrimView(prim_paths_expr="/World/envs/.*/z1/link03", name="link03_view", reset_xform_properties=False)
        self._link04 = RigidPrimView(prim_paths_expr="/World/envs/.*/z1/link04", name="link04_view", reset_xform_properties=False)
        self._link05 = RigidPrimView(prim_paths_expr="/World/envs/.*/z1/link05", name="link05_view", reset_xform_properties=False)
        self._link06 = RigidPrimView(prim_paths_expr="/World/envs/.*/z1/link06", name="link06_view", reset_xform_properties=False)
        
        self._gripperStator = RigidPrimView(prim_paths_expr="/World/envs/.*/z1/gripperStator", name="gripperStator_view", reset_xform_properties=False)
        self._gripperMover = RigidPrimView(prim_paths_expr="/World/envs/.*/z1/gripperMover", name="gripperMover_view", reset_xform_properties=False)
        

