"""
Stack task with extra decorative cubes (colors + sizes). ``cubeA`` / ``cubeB`` match the
default ``Stack`` env so ``StackPolicy`` and task logic stay the same.

Usage::

    import stack_extra_env  # registers ``StackExtraCubes``
    import robosuite as suite
    env = suite.make("StackExtraCubes", robots="Panda", ...)

Import this module before ``suite.make`` so the environment is registered.
"""

from __future__ import annotations

import numpy as np
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


@register_env
class StackExtraCubes(Stack):
    """
    Same stacking task, rewards, and ``cubeA`` / ``cubeB`` as ``Stack``; adds four
    additional blocks on the table (blue, light, dark, gray wood textures; varied sizes).

    Extra blocks expose ``*_pos`` object observations so ``test.py`` can remap them into
    ``cubeA_pos`` / ``cubeB_pos`` for repeated ``StackPolicy`` cycles (tower building).
    """

    def check_upper_on_lower(self, upper_attr: str, lower_attr: str) -> bool:
        """
        True when *upper* is resting on *lower*, gripper is not holding *upper*, and
        *upper*'s center is above *lower*'s (sparse success for non–cubeA/cubeB stacks).
        """
        upper = getattr(self, upper_attr)
        lower = getattr(self, lower_attr)
        if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=upper):
            return False
        if not self.check_contact(upper, lower):
            return False
        uid = self.sim.model.body_name2id(upper.root_body)
        lid = self.sim.model.body_name2id(lower.root_body)
        up = self.sim.data.body_xpos[uid]
        lp = self.sim.data.body_xpos[lid]
        return float(up[2]) > float(lp[2]) + 0.008

    def _setup_references(self):
        super()._setup_references()
        self.decor_blue_body_id = self.sim.model.body_name2id(self.decor_blue.root_body)
        self.decor_light_body_id = self.sim.model.body_name2id(self.decor_light.root_body)
        self.decor_dark_body_id = self.sim.model.body_name2id(self.decor_dark.root_body)
        self.decor_gray_body_id = self.sim.model.body_name2id(self.decor_gray.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables
        modality = "object"

        def _make_decor_pos_sensor(name: str, body_id: int):
            @sensor(modality=modality)
            def _decor_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[body_id])

            _decor_pos.__name__ = f"{name}_pos"
            return _decor_pos

        for name, bid in (
            ("decor_blue", self.decor_blue_body_id),
            ("decor_light", self.decor_light_body_id),
            ("decor_dark", self.decor_dark_body_id),
            ("decor_gray", self.decor_gray_body_id),
        ):
            s = _make_decor_pos_sensor(name, bid)
            observables[s.__name__] = Observable(
                name=s.__name__,
                sensor=s,
                sampling_rate=self.control_freq,
            )
        return observables

    def _load_model(self):
        super(Stack, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="lightwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="darkwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        graywood = CustomMaterial(
            texture="WoodgrainGray",
            tex_name="graywood",
            mat_name="graywood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # Identical to robosuite Stack (policy targets cubeA on cubeB)
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        # Extra blocks: varied half-extents (m) and materials; positions in obs as *_pos
        self.decor_blue = BoxObject(
            name="decor_blue",
            size_min=[0.015, 0.015, 0.015],
            size_max=[0.015, 0.015, 0.015],
            rgba=[0.2, 0.45, 0.95, 1],
            material=bluewood,
        )
        self.decor_light = BoxObject(
            name="decor_light",
            size_min=[0.023, 0.023, 0.023],
            size_max=[0.023, 0.023, 0.023],
            rgba=[0.95, 0.9, 0.75, 1],
            material=lightwood,
        )
        self.decor_dark = BoxObject(
            name="decor_dark",
            size_min=[0.014, 0.014, 0.028],
            size_max=[0.014, 0.014, 0.028],
            rgba=[0.25, 0.2, 0.18, 1],
            material=darkwood,
        )
        self.decor_gray = BoxObject(
            name="decor_gray",
            size_min=[0.021, 0.012, 0.016],
            size_max=[0.021, 0.012, 0.016],
            rgba=[0.55, 0.55, 0.58, 1],
            material=graywood,
        )

        cubes = [
            self.cubeA,
            self.cubeB,
            self.decor_blue,
            self.decor_light,
            self.decor_dark,
            self.decor_gray,
        ]

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.11, 0.11],
                y_range=[-0.11, 0.11],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )
