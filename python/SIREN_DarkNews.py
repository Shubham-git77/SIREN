# Standard python libraries
import numpy as np
import os
import datetime
import json
import ntpath
import pickle
import functools
import logging
from scipy.interpolate import LinearNDInterpolator,PchipInterpolator

# SIREN methods
from siren.interactions import DarkNewsCrossSection, DarkNewsDecay
from siren import dataclasses
from siren.dataclasses import Particle
from siren import _util
from siren.DNModelContainer import ModelContainer

# DarkNews methods
import DarkNews
from DarkNews import phase_space
from DarkNews.nuclear_tools import NuclearTarget
from DarkNews.integrands import get_decay_momenta_from_vegas_samples


resources_dir = _util.resource_package_dir()

cross_section_kwarg_keys = ["tolerance",
                            "interp_tolerance",
                            "always_interpolate"]

# Class containing all upscattering and decay modes available in DarkNews
class PyDarkNewsInteractionCollection:
    def __init__(
        self,
        table_dir=None,
        param_file=None,
        use_pickles=True,
        **kwargs,
    ):
        # Defines a series of upscattering and decay objects
        # Each derive from the respective SIREN classes

        # Separate kwargs
        model_kwargs, xs_kwargs = {}, {}
        for kw,dat in kwargs.items():
            if kw in cross_section_kwarg_keys:
                xs_kwargs[kw] = dat
            else:
                model_kwargs[kw] = dat

        # Get our model container with all ups_case and dec_case DarkNews objects
        self.models = ModelContainer(param_file, **model_kwargs)
        self.table_dir = table_dir

        # Default table_dir settings
        if self.table_dir is None:
            self.table_dir = os.path.join(
                resources_dir,
                "Processes",
                "DarkNewsTables",
                datetime.datetime.now().strftime("%Y_%m_%d__%H:%M"),
            )

        # Make the table directory where will we store cross section integrators
        table_dir_exists = False
        if os.path.exists(self.table_dir):
            # print("Directory '%s' already exists"%self.table_dir)
            table_dir_exists = True
        else:
            try:
                os.makedirs(self.table_dir, exist_ok=False)
                print("Directory '%s' created successfully" % self.table_dir)
            except OSError as error:
                print("Directory '%s' cannot be created" % self.table_dir)
                exit(0)

        if table_dir_exists:
            # Ensure that the model requested matches the model file already in the dictionary
            if param_file is not None:
                # ensure the param filename already exists
                param_filename = ntpath.basename(param_file)  # should be OS-independent
                assert os.path.isfile(os.path.join(self.table_dir, param_filename))
            # Make sure the model arguments agree
            with open(
                os.path.join(self.table_dir, "model_parameters.json"),
            ) as f:
                _model_args_dict = json.load(f)
                assert self.models.model_args_dict == _model_args_dict
        else:
            # Write a file to the directory containing infomration on the parameters used to create the model
            if param_file is not None:
                # Copy the param_file to the folder
                command = "scp " + param_file + " " + self.table_dir
                os.system(command)
            # Dump the model arguments
            with open(os.path.join(self.table_dir, "model_parameters.json"), "w") as f:
                json.dump(self.models.model_args_dict, f)

        self.GenerateCrossSections(use_pickles=use_pickles,**xs_kwargs)
        self.GenerateDecays(use_pickles=use_pickles)

    def GenerateCrossSections(self, use_pickles, **kwargs):
        # Save all unique scattering processes
        self.cross_sections = []
        for ups_key, ups_case in self.models.ups_cases.items():
            table_subdirs = "CrossSection_"
            for x in ups_key:
                if type(x) == NuclearTarget:
                    x = x.name
                table_subdirs += "%s_" % str(x)
            table_subdirs += "/"
            table_dir=os.path.join(self.table_dir, table_subdirs)
            fname = os.path.join(table_dir,"xs_object.pkl")
            if use_pickles and os.path.isfile(fname):
                with open(fname,"rb") as f:
                    xs_obj = pickle.load(f)
                    xs_obj.configure(**kwargs)
                    self.cross_sections.append(xs_obj)
            else:
                self.cross_sections.append(
                    PyDarkNewsCrossSection(
                        ups_case,
                        table_dir=table_dir,
                        **kwargs
                    )
                )

    def GenerateDecays(self, use_pickles, **kwargs):
        # Save all unique decay processes
        self.decays = []
        for dec_key, dec_case in self.models.dec_cases.items():
            table_subdirs = "Decay_"
            for x in dec_key:
                table_subdirs += "%s_" % str(x)
            table_subdirs += "/"
            table_dir=os.path.join(self.table_dir, table_subdirs)
            fname = os.path.join(table_dir,"dec_object.pkl")
            if use_pickles and os.path.isfile(fname):
                with open(fname,"rb") as f:
                    self.decays.append(pickle.load(f))
            else:
                self.decays.append(
                    PyDarkNewsDecay(
                        dec_case,
                        table_dir=table_dir,
                        **kwargs
                    )
                )

    # Save numpy arrays for total and differential cross sections
    # also pickles cross section/decay objects
    def SaveCrossSectionTables(self, fill_tables_at_exit=True):
        if not fill_tables_at_exit:
            print(
                "WARNING: Saving tables without filling PyDarkNewsCrossSection interpolation tables. Future updates to DarkNews can lead to inconsistent behavior if new entries are ever added to this table"
            )
        for cross_section in self.cross_sections:
            if fill_tables_at_exit:
                print("Filling cross section table at %s" % cross_section.table_dir)
                num = cross_section.FillInterpolationTables()
                print("Added %d points" % num)
            cross_section.SaveInterpolationTables()
            with open(os.path.join(cross_section.table_dir, "xs_object.pkl"),"wb") as f:
                pickle.dump(cross_section,f)
        for decay in self.decays:
            with open(os.path.join(decay.table_dir,"dec_object.pkl"),"wb") as f:
                pickle.dump(decay,f)

    # Fill every cross section table
    def FillCrossSectionTables(self, Emax=None):
        for cross_section in self.cross_sections:
            print("Filling cross section table at %s" % cross_section.table_dir)
            num = cross_section.FillInterpolationTables(Emax=Emax)
            print("Added %d points" % num)

    # Fill every cross section table
    def FillCrossSectionTablesAtEnergy(self, E):
        for cross_section in self.cross_sections:
            E_existing_total = np.unique(cross_section.total_cross_section_table[:, 0])
            E_existing_diff = np.unique(cross_section.differential_cross_section_table[:, 0])
            total = (E not in E_existing_total)
            diff = (E not in E_existing_diff)
            print("Filling E = %2.2f GeV for cross section table(s) at %s" %(E,cross_section.table_dir))
            num = cross_section.FillTableAtEnergy(E,total=total,diff=diff)
            print("Added %d points" % num)


# A class representing a single ups_case DarkNews class
# Only handles methods concerning the upscattering part
class PyDarkNewsCrossSection(DarkNewsCrossSection):
    def __init__(
        self,
        ups_case,  # DarkNews UpscatteringProcess instance
        table_dir=None,  # table to store
        tolerance=1e-6,  # supposed to represent machine epsilon
        interp_tolerance=5e-2,  # relative interpolation tolerance
        always_interpolate=True, # bool whether to always interpolate the total/differential cross section
    ):
        DarkNewsCrossSection.__init__(self)  # C++ constructor

        self.ups_case = ups_case
        self.tolerance = tolerance
        self.interp_tolerance = interp_tolerance
        self.table_dir = table_dir
        self.always_interpolate = always_interpolate

        # 2D table in E, sigma
        self.total_cross_section_table = np.empty((0, 2), dtype=float)
        # 3D table in E, z, dsigma/dQ2 where z = (Q2 - Q2min) / (Q2max - Q2min)
        self.differential_cross_section_table = np.empty((0, 3), dtype=float)

        if table_dir is None:
            print(
                "No table_dir specified; disabling interpolation\nWARNING: this will siginficantly slow down event generation"
            )
            return

        # Make the table directory where will we store cross section tables
        table_dir_exists = False
        if os.path.exists(self.table_dir):
            # print("Directory '%s' already exists"%self.table_dir)
            table_dir_exists = True
        else:
            try:
                os.makedirs(self.table_dir, exist_ok=False)
                print("Directory '%s' created successfully" % self.table_dir)
            except OSError as error:
                print("Directory '%s' cannot be created" % self.table_dir)
                exit(0)

        # Look in table dir and check whether total/differential xsec tables exist
        if table_dir_exists:
            total_xsec_file = os.path.join(self.table_dir, "total_cross_sections.npy")
            if os.path.exists(total_xsec_file):
                self.total_cross_section_table = np.load(total_xsec_file)
            diff_xsec_file = os.path.join(
                self.table_dir, "differential_cross_sections.npy"
            )
            if os.path.exists(diff_xsec_file):
                self.differential_cross_section_table = np.load(diff_xsec_file)

        self.configure()

    # serialization method
    def get_representation(self):
        return {"total_cross_section_table":self.total_cross_section_table,
                "differential_cross_section_table":self.differential_cross_section_table,
                "ups_case":self.ups_case,
                "tolerance":self.tolerance,
                "interp_tolerance":self.interp_tolerance,
                "table_dir":self.table_dir,
                "always_interpolate":self.always_interpolate,
                "is_configured":False
               }

    # Configure function to set up member variables
    # assumes we have defined the following:
    #   ups_case, total_cross_section_table, differential_cross_section_table,
    #   tolerance, interp_tolerance, table_dir, always_interpolate
    #   kwargs argument can be used to set any of these
    def configure(self, **kwargs):

        for k,v in kwargs.items():
            self.__setattr__(k,v)

        # Define the target particle
        # make sure protons are stored as H nuclei
        self.target_type = Particle.ParticleType(self.ups_case.nuclear_target.pdgid)
        if self.target_type==Particle.ParticleType.PPlus:
            self.target_type = Particle.ParticleType.HNucleus

        # Initialize interpolation objects
        self.total_cross_section_interpolator = None
        self.differential_cross_section_interpolator = None
        self._redefine_interpolation_objects(total=True, diff=True)
        self.is_configured = True

    # Sorts and redefines scipy interpolation objects
    def _redefine_interpolation_objects(self, total=False, diff=False):
        if total:
            if len(self.total_cross_section_table) <= 1: return
            idxs = np.argsort(
                self.total_cross_section_table[:,0]
            )
            self.total_cross_section_table = self.total_cross_section_table[idxs]
            self.total_cross_section_interpolator = PchipInterpolator(
                self.total_cross_section_table[:, 0],
                self.total_cross_section_table[:, 1]
            )
        if diff:
            if len(self.differential_cross_section_table) <= 1: return
            idxs = np.lexsort(
                (self.differential_cross_section_table[:,1],
                 self.differential_cross_section_table[:,0])
            )
            self.differential_cross_section_table = self.differential_cross_section_table[idxs]
            # If we only have two energy points, don't try to construct interpolator
            if len(np.unique(self.differential_cross_section_table[:, 0])) <= 2: return
            self.differential_cross_section_interpolator = (
                LinearNDInterpolator(
                    self.differential_cross_section_table[:, :2],
                    self.differential_cross_section_table[:, 2],
                    rescale=True,
                )
            )

    # Check whether we have close-enough entries in the intrepolation tables
    def _interpolation_flags(self, inputs, mode):
        #
        # returns UseSinglePoint,Interpolate,closest_idx
        # UseSinglePoint: whether to use a single point in table
        # Interpolate: whether to interpolate bewteen different points
        # closest_idx: index of closest point in table (for UseSinglePoint)

        # Determine which table we are using
        if mode == "total":
            interp_table = self.total_cross_section_table
        elif mode == "differential":
            interp_table = self.differential_cross_section_table
        else:
            print("Invalid interpolation table mode %s" % mode)
            exit(0)

        # first check if we have saved table points already
        if len(interp_table) == 0:
            return False, False, -1

        # bools to keep track of whether to use a single point or interpolate
        UseSinglePoint = False
        Interpolate = True
        # order events by the relative difference
        rel_diff = np.abs((interp_table[:, :-1] - inputs) / inputs)
        rel_diff_length = np.sqrt(np.sum(rel_diff**2,axis=-1))
        closest_idx_abs = np.argmin(rel_diff_length,axis=-1)
        # First check whether we have a close-enough single point
        if np.all(np.abs(rel_diff[closest_idx_abs]) < self.tolerance):
            UseSinglePoint = True
        # Ensure we have enough points to interpolate
        if len(interp_table) < len(inputs)+1:
            Interpolate = False
        # Require that we have at least len(inputs)+1 close points to interpolate
        else:
            close = np.all(rel_diff<self.interp_tolerance,axis=-1)
            if sum(close) < len(inputs)+1:
                Interpolate = False
        return UseSinglePoint, Interpolate, closest_idx_abs

    # return entries in interpolation table if we have inputs
    def _query_interpolation_table(self, inputs, mode):
        #
        # returns:
        # 0 if we are not close enough to any points in the interpolation table
        # otherwise, returns the desired interpolated value

        # First make sure we are configured
        self._ensure_configured()

        # Determine which table we are using
        if mode == "total":
            interp_table = self.total_cross_section_table
            interpolator = self.total_cross_section_interpolator
        elif mode == "differential":
            interp_table = self.differential_cross_section_table
            interpolator = self.differential_cross_section_interpolator
        else:
            print("Invalid interpolation table mode %s" % mode)
            exit(0)

        if self.always_interpolate:
            # check if energy is within table range

            if interpolator is None or inputs[0] > interp_table[-1,0]:
                print("Requested interpolation at %2.2f GeV. Either this is above the table boundary or the interpolator doesn't yet exist. Filling %s table"%(inputs[0],mode))
                n = self.FillInterpolationTables(total=(mode=="total"),
                                                 diff=(mode=="differential"),
                                                 Emax = (1+self.interp_tolerance)*inputs[0])
                print("Added %d points"%n)
                if mode == "total": interpolator = self.total_cross_section_interpolator
                elif mode== "differential": interpolator = self.differential_cross_section_interpolator
            elif inputs[0] < interp_table[0,0]:
                print("Requested interpolation at %2.2f GeV below table boundary. Requring calculation"%inputs[0])
                return 0
            val = max(0,interpolator(inputs))
            if val<0:
                print("WARNING: negative interpolated value for %s-%s %s cross section at,"%(self.ups_case.nuclear_target.name,
                                                                                             self.ups_case.scattering_regime,
                                                                                             mode),inputs)
            return val

        UseSinglePoint, Interpolate, closest_idx = self._interpolation_flags(
            inputs, mode
        )

        if UseSinglePoint:
            if closest_idx<0:
                print("Trying to use a single table point, but no closest idx found. Exiting...")
                exit(0)
            return interp_table[closest_idx, -1]
        elif Interpolate:
            return interpolator(inputs)
        else:
            return -1

    def FillTableAtEnergy(self, E, total=True, diff=True, factor=0.8):
        num_added_points = 0
        if total:
            xsec = self.ups_case.total_xsec(E)
            self.total_cross_section_table = np.append(
                self.total_cross_section_table, [[E, xsec]], axis=0
            )
            num_added_points+=1
        if diff:
            interaction = dataclasses.InteractionRecord()
            interaction.signature.primary_type = self.GetPossiblePrimaries()[
                0
            ]  # only one primary
            interaction.signature.target_type = self.GetPossibleTargets()[
                0
            ]  # only one target
            interaction.target_mass = self.ups_case.MA
            interaction.primary_momentum = [E, 0, 0, 0]
            zmin, zmax = self.tolerance, 1
            Q2min = self.Q2Min(interaction)
            Q2max = self.Q2Max(interaction)
            z = zmin
            while z < zmax:
                Q2 = Q2min + z * (Q2max - Q2min)
                dxsec = float(np.real(self.ups_case.diff_xsec_Q2(E, Q2)))
                self.differential_cross_section_table = np.append(
                    self.differential_cross_section_table,
                    [[E, z, dxsec]],
                    axis=0,
                )
                num_added_points += 1
                z *= (1 + factor*self.interp_tolerance)
        self._redefine_interpolation_objects(total=total, diff=diff)
        return num_added_points


    # Fills the total and differential cross section tables within interp_tolerance
    def FillInterpolationTables(self, total=True, diff=True, factor=0.8, Emax=None):
        increment_factor = 0.5*factor * self.interp_tolerance
        Emin = (1.0 + self.tolerance) * self.ups_case.Ethreshold
        if Emax is None:
            if (len(self.total_cross_section_table) +
                len(self.differential_cross_section_table)) <=0:
                return 0
            Emax = max(np.max([0] + list(self.total_cross_section_table[:, 0])),
                       np.max([0] + list(self.differential_cross_section_table[:, 0])))
        num_added_points = 0
        E = Emin
        E_existing_total = np.unique(self.total_cross_section_table[:, 0])
        E_existing_diff = np.unique(self.differential_cross_section_table[:, 0])
        while E < Emax:
            # sample more coarsely past 1.5*threshold
            if E > 1.5*self.ups_case.Ethreshold:
                increment_factor = factor * self.interp_tolerance
            n = self.FillTableAtEnergy(E,
                                       total=(total and (E not in E_existing_total)),
                                       diff=(diff and (E not in E_existing_diff)),
                                       factor=factor)
            num_added_points += n
            E *= (1 + increment_factor)
        self._redefine_interpolation_objects(total=total, diff=diff)
        return num_added_points

    # Saves the tables for the scipy interpolation objects
    def SaveInterpolationTables(self, total=True, diff=True):
        if total:
            self._redefine_interpolation_objects(total=True)
            with open(
                os.path.join(self.table_dir, "total_cross_sections.npy"), "wb"
            ) as f:
                np.save(f, self.total_cross_section_table)
        if diff:
            self._redefine_interpolation_objects(diff=True)
            with open(
                os.path.join(self.table_dir, "differential_cross_sections.npy"), "wb"
            ) as f:
                np.save(f, self.differential_cross_section_table)

    def GetPossiblePrimaries(self):
        return [Particle.ParticleType(self.ups_case.nu_projectile.pdgid)]

    def _ensure_configured(self):
        if not self.is_configured:
            self.configure()

    def GetPossibleTargetsFromPrimary(self, primary_type):
        self._ensure_configured()
        if Particle.ParticleType(self.ups_case.nu_projectile.pdgid) == primary_type:
            return [self.target_type]
        return []

    def GetPossibleTargets(self):
        self._ensure_configured()
        return [self.target_type]

    def GetPossibleSignatures(self):
        self._ensure_configured()
        signature = dataclasses.InteractionSignature()
        signature.primary_type = Particle.ParticleType(
            self.ups_case.nu_projectile.pdgid
        )
        signature.target_type = self.target_type
        signature.secondary_types = []
        signature.secondary_types.append(
            Particle.ParticleType(self.ups_case.nu_upscattered.pdgid)
        )
        signature.secondary_types.append(
            self.target_type
        )
        return [signature]

    def GetPossibleSignaturesFromParents(self, primary_type, target_type):
        if (
            Particle.ParticleType(self.ups_case.nu_projectile.pdgid) == primary_type
        ) and (
            (self.target_type == target_type)
        ):
            signature = dataclasses.InteractionSignature()
            signature.primary_type = Particle.ParticleType(
                self.ups_case.nu_projectile.pdgid
            )
            signature.target_type = self.target_type
            secondary_types = []
            secondary_types.append(
                Particle.ParticleType(self.ups_case.nu_upscattered.pdgid)
            )
            secondary_types.append(
                Particle.ParticleType(self.ups_case.nuclear_target.pdgid)
            )
            signature.secondary_types = secondary_types
            return [signature]
        return []

    def DifferentialCrossSection(self, arg1, target=None, energy=None, Q2=None):
        if type(arg1) == dataclasses.InteractionRecord:
            interaction = arg1
            # Calculate Q2 assuming we are in the target rest frame
            m1sq = max(0.0, interaction.primary_momentum[0] ** 2 - np.sum(
                [p**2 for p in interaction.primary_momentum[1:]]
            ))  # clamp: E²-|p|² can be slightly negative due to float precision
            m3sq = max(0.0, interaction.secondary_momenta[0][0] ** 2 - np.sum(
                [p**2 for p in interaction.secondary_momenta[0][1:]]
            ))  # clamp: same reason
            p1p3 = interaction.primary_momentum[0] * interaction.secondary_momenta[0][
                0
            ] - np.sum(
                p1 * p3
                for p1, p3 in zip(
                    interaction.primary_momentum[1:],
                    interaction.secondary_momenta[0][1:],
                )
            )
            Q2 = max(0.0, -(m1sq + m3sq - 2 * p1p3))  # clamp: numerical noise can make Q2 slightly negative
            energy = interaction.primary_momentum[0]
        else:
            primary = arg1
            interaction = dataclasses.InteractionRecord()
            interaction.signature.primary_type = primary
            interaction.signature.target_type = target
            interaction.primary_momentum = [energy, 0, 0, 0]
            interaction.target_mass = self.ups_case.MA
        if interaction.signature.primary_type != Particle.ParticleType(self.ups_case.nu_projectile.pdgid):
            return 0
        if interaction.primary_momentum[0] < self.InteractionThreshold(interaction):
            return 0
        Q2min = self.Q2Min(interaction)
        Q2max = self.Q2Max(interaction)
        if Q2 < Q2min or Q2 > Q2max:
            return 0
        z = (Q2 - Q2min) / (Q2max - Q2min)

        if self.always_interpolate:
            # Check if we can interpolate
            val = self._query_interpolation_table([energy, z], mode="differential")
            if val >= 0:
                # we have recovered the differential cross section from the interpolation table
                return val

        # If we have reached this block, we must compute the differential cross section using DarkNews
        dxsec = float(np.real(self.ups_case.diff_xsec_Q2(energy, Q2)))
        return dxsec

    def TargetMass(self, target_type):
        target_mass = self.ups_case.MA
        return target_mass

    def SecondaryMasses(self, secondary_types):
        secondary_masses = []
        secondary_masses.append(self.ups_case.m_ups)
        secondary_masses.append(self.ups_case.MA)
        return secondary_masses

    def SecondaryHelicities(self, record):
        secondary_helicities = []
        secondary_helicities.append(
            self.ups_case.h_upscattered * record.primary_helicity
        )
        secondary_helicities.append(record.target_helicity)
        self.h_ups = self.ups_case.m_ups
        self.h_target = self.ups_case.MA
        return secondary_helicities

    def TotalCrossSection(self, arg1, energy=None, target=None):
        # Handle overloaded arguments
        if type(arg1) == dataclasses.InteractionRecord:
            primary = arg1.signature.primary_type
            energy = arg1.primary_momentum[0]
            target = arg1.signature.target_type
        elif energy is not None and target is not None:
            primary = arg1
        else:
            print("Incorrect function call to TotalCrossSection!")
            exit(0)
        if int(primary) != self.ups_case.nu_projectile:
            return 0
        interaction = dataclasses.InteractionRecord()
        interaction.signature.primary_type = primary
        interaction.signature.target_type = target
        interaction.primary_momentum[0] = energy
        if energy < self.InteractionThreshold(interaction):
            #print("Python: energy %2.2f < self.InteractionThreshold(interaction) %2.2f"%(energy,self.InteractionThreshold(interaction)))
            return 0

        # Check if we can interpolate
        val = self._query_interpolation_table([energy], mode="total")
        if val >= 0:
            # we have recovered the cross section from the interpolation table
            return val

        # If we have reached this block, we must compute the cross section using DarkNews
        xsec = float(np.real(self.ups_case.total_xsec(energy)))
        self.total_cross_section_table = np.append(
            self.total_cross_section_table, [[energy, xsec]], axis=0
        )
        self._redefine_interpolation_objects(total=True)
        return xsec

    def InteractionThreshold(self, interaction):
        return self.ups_case.Ethreshold

    def Q2Min(self, interaction):
        return phase_space.upscattering_Q2min(
            interaction.primary_momentum[0],
            self.ups_case.m_ups,
            self.ups_case.MA,
        )

    def Q2Max(self, interaction):
        return phase_space.upscattering_Q2max(
            interaction.primary_momentum[0],
            self.ups_case.m_ups,
            self.ups_case.MA,
        )
        

import scipy.integrate as _scipy_integrate


# ─────────────────────────────────────────────────────────────────────────────
# Minimal particle / target stubs
# Must implement __eq__(int) because SIREN_DarkNews.py line 618 does:
#   if int(primary) != self.ups_case.nu_projectile:
# ─────────────────────────────────────────────────────────────────────────────

class _PDGParticle:
    """
    Lightweight particle stub that satisfies the duck-typed ups_case interface
    expected by PyDarkNewsCrossSection.

    Critical: __eq__ must accept bare ints so that line 618
        int(primary) != self.ups_case.nu_projectile
    evaluates correctly.
    """
    def __init__(self, pdgid: int, mass: float = 0.0, name: str = ""):
        self.pdgid = pdgid
        self.mass  = mass
        self.name  = name

    # ── comparison ──────────────────────────────────────────────────────────
    def __eq__(self, other):
        if isinstance(other, int):
            return self.pdgid == other
        return self.pdgid == getattr(other, "pdgid", None)

    def __ne__(self, other):
        return not self.__eq__(other)

    # int(particle) → pdgid  (used by line 618)
    def __int__(self):
        return self.pdgid

    def __repr__(self):
        return f"_PDGParticle(pdgid={self.pdgid}, name={self.name!r})"


class _NuclearTarget:

    def __init__(
        self,
        pdgid: int,
        mass: float,
        name: str,
        scattering_regime: str = "coherent",
    ):
        self.pdgid             = pdgid
        self.mass              = mass
        self.name              = name
        self.scattering_regime = scattering_regime

    def __repr__(self):
        return f"_NuclearTarget(name={self.name!r}, pdgid={self.pdgid})"


class VectorPortalUpsCase:

    def __init__(
        self,
        m_chi: float,
        m_chi_prime: float,
        m_V: float,
        g_D: float,
        epsilon: float,
        *,
        pdgid_chi: int         = 5917,
        pdgid_chi_prime: int   = 5918,
        nuclear_pdgid: int     = 1000060120,
        nuclear_mass: float    = 11.178,      # C12 in GeV
        nuclear_name: str      = "C12",
        scattering_regime: str = "coherent",
        A: int                 = 12,
    ):

        self.nu_projectile  = _PDGParticle(pdgid_chi,       m_chi,       "chi")
        self.nu_upscattered = _PDGParticle(pdgid_chi_prime, m_chi_prime, "chi_prime")
        self.nuclear_target = _NuclearTarget(
            nuclear_pdgid, nuclear_mass, nuclear_name, scattering_regime
        )

        self.MA              = nuclear_mass    # target mass [GeV]
        self.m_ups           = m_chi_prime     # upscattered mass [GeV]
        self.m_chi           = m_chi
        self.m_V             = m_V
        self.g_D             = g_D
        self.epsilon         = epsilon
        self.A               = A

        self.h_upscattered     = 1             # helicity convention (scalar χ → +1)
        self.scattering_regime = scattering_regime
        self.Ethreshold = self._compute_threshold()

    def _compute_threshold(self) -> float:
        """Minimum lab-frame χ energy for χ N → χ' N kinematics."""
        m1, m3, M = self.m_chi, self.m_ups, self.MA
        return ((m3 + M) ** 2 - m1 ** 2 - M ** 2) / (2.0 * M)

    def _helm_F2(self, Q2: float) -> float:

        Q    = np.sqrt(max(Q2, 0.0))          # GeV
        Qfm  = Q / 0.197327                   # fm⁻¹
        s    = 0.9                             # fm  (skin thickness)
        r0sq = max((1.2 * self.A ** (1.0 / 3.0)) ** 2 - 5.0 * s ** 2, 0.0)
        Qr   = Qfm * np.sqrt(r0sq)
        # Spherical Bessel j₁(x)/x = (sin x − x cos x)/x³
        if Qr < 1.0e-6:
            j1_over_Qr = 1.0 / 3.0
        else:
            j1_over_Qr = (np.sin(Qr) - Qr * np.cos(Qr)) / Qr ** 3
        return (3.0 * j1_over_Qr) ** 2 * np.exp(-(Qfm * s) ** 2)

    def _dsigma_dQ2(self, E: float, Q2: float) -> float:

        m1, m3       = self.m_chi, self.m_ups
        M, mV        = self.MA, self.m_V
        gD, eps      = self.g_D, self.epsilon
        alpha        = 1.0 / 137.036          # fine-structure constant
        GeV2_to_cm2  = 3.8938e-28             # ℏ²c² in cm²·GeV²

        s    = m1 ** 2 + M ** 2 + 2.0 * M * E
        flux = (s - M ** 2) ** 2              # lab-frame flux squared

        # Four-momentum transfer squared (spacelike → Q² > 0)
        Delta_m2 = m3 ** 2 - m1 ** 2
        numerator = 2.0 * M ** 2 * (2.0 * E * M - Q2 - Delta_m2)

        # Guard against unphysical kinematics
        if numerator <= 0.0 or flux <= 0.0:
            return 0.0

        propagator = 1.0 / (Q2 + mV ** 2) ** 2
        M2 = gD ** 2 * 4.0 * np.pi * alpha * eps ** 2 * numerator * propagator

        dsig = M2 * self._helm_F2(Q2) / (16.0 * np.pi * flux)
        return max(0.0, dsig) * GeV2_to_cm2


    def diff_xsec_Q2(self, E: float, Q2: float) -> np.ndarray:
       
        return np.array(self._dsigma_dQ2(E, Q2))

    def total_xsec(self, E: float) -> float:
 
        Q2min = phase_space.upscattering_Q2min(E, self.m_ups, self.MA)
        Q2max = phase_space.upscattering_Q2max(E, self.m_ups, self.MA)
        if Q2max <= Q2min:
            return 0.0
        result, _ = _scipy_integrate.quad(
            lambda q2: self._dsigma_dQ2(E, q2),
            Q2min,
            Q2max,
            limit=80,
            epsrel=1e-4,
        )
        return max(0.0, result)

    def __repr__(self):
        return (
            f"VectorPortalUpsCase("
            f"m_chi={self.m_chi}, m_chi'={self.m_ups}, "
            f"m_V={self.m_V}, g_D={self.g_D}, ε={self.epsilon})"
        )               

class PyDarkNewsDecay(DarkNewsDecay):
    def __init__(self, dec_case, table_dir=None):
        DarkNewsDecay.__init__(self)  # C++ constructor
        self.dec_case = dec_case
        self.table_dir = table_dir

        # Some variables for storing the decay phase space integrator
        self.decay_integrator = None
        self.decay_norm = None
        self.PS_samples = None
        self.PS_weights = None
        self.PS_weights_CDF = None
        self.total_width = None

        if table_dir is None:
            print(
                "No table_dir specified; will sample from new VEGAS integrator for each decay"
            )
            print("WARNING: this will siginficantly slow down event generation")
            return

        # Make the table directory where will we store cross section integrators
        table_dir_exists = False
        if os.path.exists(self.table_dir):
            # print("Directory '%s' already exists"%self.table_dir)
            table_dir_exists = True
        else:
            try:
                os.makedirs(self.table_dir, exist_ok=False)
                print("Directory '%s' created successfully" % self.table_dir)
            except OSError as error:
                print("Directory '%s' cannot be created" % self.table_dir)
                exit(0)

        if table_dir_exists:
            self.SetIntegratorAndNorm()

    def get_representation(self):
        return {"decay_integrator":self.decay_integrator,
                "decay_norm":self.decay_norm,
                "dec_case":self.dec_case,
                "PS_samples":self.PS_samples,
                "PS_weights":self.PS_weights,
                "PS_weights_CDF":self.PS_weights_CDF,
                "total_width":self.total_width,
                "table_dir":self.table_dir
               }

    def SetIntegratorAndNorm(self):
        # Try to find the decay integrator
        int_file = os.path.join(self.table_dir, "decay_integrator.pkl")
        if os.path.isfile(int_file):
            with open(int_file, "rb") as ifile:
                _, self.decay_integrator = pickle.load(ifile)
        # Try to find the normalization information
        norm_file = os.path.join(self.table_dir, "decay_norm.json")
        if os.path.isfile(norm_file):
            with open(
                norm_file,
            ) as nfile:
                self.decay_norm = json.load(nfile)

    def GetPossibleSignatures(self):
        signature = dataclasses.InteractionSignature()
        signature.primary_type = Particle.ParticleType(self.dec_case.nu_parent.pdgid)
        signature.target_type = Particle.ParticleType.Decay
        secondary_types = []
        secondary_types.append(Particle.ParticleType(self.dec_case.nu_daughter.pdgid))
        for secondary in self.dec_case.secondaries:
            secondary_types.append(Particle.ParticleType(secondary.pdgid))
        signature.secondary_types = secondary_types
        return [signature]

    def GetPossibleSignaturesFromParent(self, primary_type):
        if Particle.ParticleType(self.dec_case.nu_parent.pdgid) == primary_type:
            signature = dataclasses.InteractionSignature()
            signature.primary_type = Particle.ParticleType(
                self.dec_case.nu_parent.pdgid
            )
            signature.target_type = Particle.ParticleType.Decay
            secondary_types = []
            secondary_types.append(
                Particle.ParticleType(self.dec_case.nu_daughter.pdgid)
            )
            for secondary in self.dec_case.secondaries:
                secondary_types.append(Particle.ParticleType(secondary.pdgid))
            signature.secondary_types = secondary_types
            return [signature]
        return []

    def DifferentialDecayWidth(self, record):
        # Momentum variables of HNL necessary for calculating decay phase space
        PN = np.array(record.primary_momentum)

        if type(self.dec_case) == DarkNews.processes.FermionSinglePhotonDecay:
            gamma_idx = 0
            for secondary in record.signature.secondary_types:
                if secondary == dataclasses.Particle.ParticleType.Gamma:
                    break
                gamma_idx += 1
            if gamma_idx >= len(record.signature.secondary_types):
                print("No gamma found in the list of secondaries!")
                exit(0)

            Pgamma = np.array(record.secondary_momenta[gamma_idx])
            momenta = np.expand_dims(PN, 0), np.expand_dims(Pgamma, 0)

        elif type(self.dec_case) == DarkNews.processes.FermionDileptonDecay:
            lepminus_idx = -1
            lepplus_idx = -1
            nu_idx = -1
            for idx, secondary in enumerate(record.signature.secondary_types):
                if secondary in [
                    dataclasses.Particle.ParticleType.EMinus,
                    dataclasses.Particle.ParticleType.MuMinus,
                    dataclasses.Particle.ParticleType.TauMinus,
                ]:
                    lepminus_idx = idx
                elif secondary in [
                    dataclasses.Particle.ParticleType.EPlus,
                    dataclasses.Particle.ParticleType.MuPlus,
                    dataclasses.Particle.ParticleType.TauPlus,
                ]:
                    lepplus_idx = idx
                else:
                    nu_idx = idx
            if -1 in [lepminus_idx, lepplus_idx, nu_idx]:
                print("Couldn't find two leptons and a neutrino in the final state!")
                exit(0)
            Pnu = np.array(record.secondary_momenta[nu_idx])
            Plepminus = np.array(record.secondary_momenta[lepminus_idx])
            Plepplus = np.array(record.secondary_momenta[lepplus_idx])
            momenta = (
                np.expand_dims(PN, 0),
                np.expand_dims(Plepminus, 0),
                np.expand_dims(Plepplus, 0),
                np.expand_dims(Pnu, 0),
            )
        else:
            print("%s is not a valid decay class type!" % type(self.dec_case))
            exit(0)
        return self.dec_case.differential_width(momenta)

    def TotalDecayWidth(self, arg1):
        if type(arg1) == dataclasses.InteractionRecord:
            primary = arg1.signature.primary_type
        elif type(arg1) == dataclasses.Particle.ParticleType:
            primary = arg1
        else:
            print("Incorrect function call to TotalDecayWidth!")
            exit(0)
        if int(primary) != self.dec_case.nu_parent:
            return 0
        if self.total_width is None:
            # Need to set the total width
            if type(self.dec_case) == DarkNews.processes.FermionDileptonDecay and (
                self.dec_case.vector_off_shell and self.dec_case.scalar_off_shell
            ):
                # total width calculation requires evaluating an integral
                if self.decay_integrator is None or self.decay_norm is None:
                    # We need to initialize a new VEGAS integrator in DarkNews
                    int_file = os.path.join(self.table_dir, "decay_integrator.pkl")
                    norm_file = os.path.join(self.table_dir, "decay_norm.json")
                    self.total_width = self.dec_case.total_width(
                        savefile_norm=norm_file, savefile_dec=int_file
                    )
                    self.SetIntegratorAndNorm()
                else:
                    self.total_width = (
                        self.decay_integrator["diff_decay_rate_0"].mean
                        * self.decay_norm["diff_decay_rate_0"]
                    )
            else:
                self.total_width = self.dec_case.total_width()
        return self.total_width

    def TotalDecayWidthForFinalState(self, record):
        sig = self.GetPossibleSignatures()[0]
        if (
            (record.signature.primary_type != sig.primary_type)
            or (record.signature.target_type != sig.target_type)
            or (len(record.signature.secondary_types) != len(sig.secondary_types))
            or (
                np.any(
                    [
                        record.signature.secondary_types[i] != sig.secondary_types[i]
                        for i in range(len(sig.secondary_types))
                    ]
                )
            )
        ):
            return 0
        ret = self.dec_case.total_width()
        return ret

    def DensityVariables(self):
        if type(self.dec_case) == DarkNews.processes.FermionSinglePhotonDecay:
            return "cost"
        elif type(self.dec_case) == DarkNews.processes.FermionDileptonDecay:
            if self.dec_case.vector_on_shell and self.dec_case.scalar_on_shell:
                print("Can't have both the scalar and vector on shell")
                exit(0)
            elif (self.dec_case.vector_on_shell and self.dec_case.scalar_off_shell) or (
                self.dec_case.vector_off_shell and self.dec_case.scalar_on_shell
            ):
                return "cost"
            elif self.dec_case.vector_off_shell and self.dec_case.scalar_off_shell:
                return "t,u,c3,phi34"
        else:
            print("%s is not a valid decay class type!" % type(self.dec_case))
            exit(0)
        return ""

    def GetPSSample(self, random):
        # Make the PS weight CDF if that hasn't been done
        if self.PS_weights_CDF is None:
            self.PS_weights_CDF = np.cumsum(self.PS_weights)

        # Random number to determine
        x = random.Uniform(0, self.PS_weights_CDF[-1])

        # find first instance of a CDF entry greater than x
        PSidx = np.argmax(x - self.PS_weights_CDF <= 0)
        return self.PS_samples[:, PSidx]

    def SampleRecordFromDarkNews(self, record, random):
        # First, make sure we have PS samples and weights
        if self.PS_samples is None or self.PS_weights is None:
            # We need to generate new PS samples
            if self.decay_integrator is None or self.decay_norm is None:
                # We need to initialize a new VEGAS integrator in DarkNews
                int_file = os.path.join(self.table_dir, "decay_integrator.pkl")
                norm_file = os.path.join(self.table_dir, "decay_norm.json")
                self.PS_samples, PS_weights_dict = self.dec_case.SamplePS(
                    savefile_norm=norm_file, savefile_dec=int_file
                )
                self.PS_weights = PS_weights_dict["diff_decay_rate_0"]
                self.SetIntegratorAndNorm()
            else:
                # We already have an integrator, we just need new PS samples
                self.PS_samples, PS_weights_dict = self.dec_case.SamplePS(
                    existing_integrator=self.decay_integrator
                )
                self.PS_weights = PS_weights_dict["diff_decay_rate_0"]

        # Now we must sample an PS point on the hypercube
        PS = self.GetPSSample(random)

        # Find the four-momenta associated with this point
        # Expand dims required to call DarkNews function on signle sample
        four_momenta = get_decay_momenta_from_vegas_samples(
            np.expand_dims(PS, 0),
            self.dec_case,
            np.expand_dims(np.array(record.primary_momentum), 0),
        )

        secondaries = record.GetSecondaryParticleRecords()

        if type(self.dec_case) == DarkNews.processes.FermionSinglePhotonDecay:
            gamma_idx = 0
            for secondary in record.signature.secondary_types:
                if secondary == dataclasses.Particle.ParticleType.Gamma:
                    break
                gamma_idx += 1
            if gamma_idx >= len(record.signature.secondary_types):
                print("No gamma found in the list of secondaries!")
                exit(0)
            nu_idx = 1 - gamma_idx
            secondaries[gamma_idx].four_momentum = np.squeeze(four_momenta["P_decay_photon"])
            secondaries[gamma_idx].mass = 0
            secondaries[nu_idx].four_momentum = np.squeeze(four_momenta["P_decay_N_daughter"])
            secondaries[nu_idx].mass = 0

        elif type(self.dec_case) == DarkNews.processes.FermionDileptonDecay:
            lepminus_idx = -1
            lepplus_idx = -1
            nu_idx = -1
            for idx, secondary in enumerate(record.signature.secondary_types):
                if secondary in [
                    dataclasses.Particle.ParticleType.EMinus,
                    dataclasses.Particle.ParticleType.MuMinus,
                    dataclasses.Particle.ParticleType.TauMinus,
                ]:
                    lepminus_idx = idx
                elif secondary in [
                    dataclasses.Particle.ParticleType.EPlus,
                    dataclasses.Particle.ParticleType.MuPlus,
                    dataclasses.Particle.ParticleType.TauPlus,
                ]:
                    lepplus_idx = idx
                else:
                    nu_idx = idx
            if -1 in [lepminus_idx, lepplus_idx, nu_idx]:
                print([lepminus_idx, lepplus_idx, nu_idx])
                print(record.signature.secondary_types)
                print("Couldn't find two leptons and a neutrino in the final state!")
                exit(0)
            secondaries[lepminus_idx].four_momentum = (
                np.squeeze(four_momenta["P_decay_ell_minus"])
            )
            secondaries[lepplus_idx].four_momentum = (
                np.squeeze(four_momenta["P_decay_ell_plus"])
            )
            secondaries[nu_idx].four_momentum = (
                np.squeeze(four_momenta["P_decay_N_daughter"])
            )
        return record


class DarkPrimakoffUpsCase:
    """
    ups_case for  φ(k) + N(p) → γ(k') + N(p')  via t-channel Z'.

    Physics: arXiv:2110.11944 Appendix C, Eqs. (C2)-(C3).

    Matrix element squared (scalar φ, same formula as pseudoscalar with m_a→m_φ):

        |M|² = g²_n λ² / t  ×
               [2m²_N(m²_φ − 2s − t) + 2m⁴_N − 2m²_φ(s+t) + m⁴_φ + 2s² + 2st + t²]
               ─────────────────────────────────────────────────────────────────────────
                                     2(t − m²_Z')²

    where t < 0 is the spacelike Mandelstam variable (we use t = −Q² < 0 here,
    consistent with Appendix C's sign convention).

    Differential cross section (Eq. C3):

        dσ/dt = Z² / [16π λ(s, m²_φ, m²_N)] × |M|² × F²_N(t)

    where λ(s,m²_φ,m²_N) = [s − (m_φ+m_N)²][s − (m_φ−m_N)²] is the
    Källén flux factor and F_N is the Helm nuclear form factor.

    PyDarkNewsCrossSection interface attributes accessed:
        .nu_projectile   (_PDGParticle, pdgid=5902 for φ)
        .nu_upscattered  (_PDGParticle, pdgid=22 for γ)
        .nuclear_target  (_NuclearTarget)
        .MA, .m_ups, .h_upscattered, .Ethreshold, .scattering_regime
        .diff_xsec_Q2(E, Q2)  →  np.ndarray scalar
        .total_xsec(E)        →  float

    Note on variable naming: PyDarkNewsCrossSection uses Q2 as its integration
    variable and calls phase_space.upscattering_Q2min/max. For this process
    Q2 ≡ −t (positive, spacelike), so we convert internally: t = −Q2.

    Benchmark (Table I, scalar scenario):
        m_Z' = 49 MeV,  m_φ = 1 MeV,  coupling product g_μ g_n λ = 2.2e-8 MeV⁻¹
    Here we expose g_n (quark/nuclear coupling) and lam (the λ coupling) separately
    since g_μ enters only in the production flux, not the detector cross section.
    """

    def __init__(
        self,
        m_phi: float,         # scalar φ mass [GeV]
        m_Zprime: float,      # Z' mediator mass [GeV]
        g_n: float,           # Z' nuclear coupling  (dimensionless)
        lam: float,           # φ-Z'-γ coupling  [GeV⁻¹]
        *,
        pdgid_phi: int         = 5902,
        nuclear_pdgid: int     = 1000060120,   # C12
        nuclear_mass: float    = 11.178,       # GeV
        nuclear_name: str      = "C12",
        Z: int                 = 6,            # atomic number for form factor prefactor
        A: int                 = 12,           # mass number for Helm FF
        scattering_regime: str = "coherent",
    ):
        # ── particle stubs ──────────────────────────────────────────────────
        self.nu_projectile  = _PDGParticle(pdgid_phi, m_phi,  "phi")
        self.nu_upscattered = _PDGParticle(22,        0.0,    "gamma")   # photon
        self.nuclear_target = _NuclearTarget(
            nuclear_pdgid, nuclear_mass, nuclear_name, scattering_regime
        )

        # ── physics parameters ───────────────────────────────────────────────
        self.m_phi           = m_phi
        self.m_Zprime        = m_Zprime
        self.g_n             = g_n
        self.lam             = lam            # [GeV⁻¹]
        self.MA              = nuclear_mass
        self.m_ups           = 0.0            # outgoing photon mass
        self.Z               = Z
        self.A               = A
        self.h_upscattered   = 0              # photon, helicity undefined → 0
        self.scattering_regime = scattering_regime

        # ── kinematic threshold ──────────────────────────────────────────────
        # Minimum φ lab energy for φ N → γ N:
        # s_min = (m_φ + m_N)²  →  E_thr = [(m_φ + m_N)² − m_φ² − m_N²] / 2m_N
        self.Ethreshold = (
            (m_phi + nuclear_mass) ** 2 - m_phi ** 2 - nuclear_mass ** 2
        ) / (2.0 * nuclear_mass)

        # ── unit conversion ──────────────────────────────────────────────────
        self._GeV2_to_cm2 = 3.8938e-28        # ℏ²c² in cm²·GeV²

    # ── Helm form factor ─────────────────────────────────────────────────────

    def _helm_F2(self, Q2: float) -> float:

        Q   = np.sqrt(max(Q2, 0.0))
        Qfm = Q / 0.197327
        s   = 0.9
        r0sq = max((1.2 * self.A ** (1.0 / 3.0)) ** 2 - 5.0 * s ** 2, 0.0)
        Qr  = Qfm * np.sqrt(r0sq)
        if Qr < 1e-6:
            j1_over_Qr = 1.0 / 3.0
        else:
            j1_over_Qr = (np.sin(Qr) - Qr * np.cos(Qr)) / Qr ** 3
        return (3.0 * j1_over_Qr) ** 2 * np.exp(-(Qfm * s) ** 2)

    # ── matrix element squared  (Eq. C2, scalar case) ────────────────────────

    def _M2(self, s: float, t: float) -> float:
        """
        |M|²  for  φ N → γ N  (Eq. C2 with m_a → m_φ).

        t is the Mandelstam variable, t < 0  (spacelike, following paper sign).
        """
        mN  = self.MA
        mphi = self.m_phi
        mZp  = self.m_Zprime
        gn   = self.g_n
        lam  = self.lam

        # Numerator of Eq. C2
        num = (
            2.0 * mN**2 * (mphi**2 - 2.0*s - t)
            + 2.0 * mN**4
            - 2.0 * mphi**2 * (s + t)
            + mphi**4
            + 2.0 * s**2
            + 2.0 * s * t
            + t**2
        )

        # Propagator denominator: (t − m²_Z')²
        prop_denom = (t - mZp**2) ** 2

        if prop_denom <= 0.0:
            return 0.0

        return gn**2 * lam**2 * num / (2.0 * (-t) * prop_denom)

    # ── differential cross section ────────────────────────────────────────────

    def _dsigma_dt(self, E: float, t: float) -> float:
        """
        dσ/dt  [cm²/GeV²]  in the nucleus rest frame.  (Eq. C3)

        t < 0 (spacelike).  Returns 0 outside the physical region.
        """
        mN   = self.MA
        mphi = self.m_phi

        s = mphi**2 + mN**2 + 2.0 * mN * E

        # Källén flux factor  λ(s, m²_φ, m²_N) = [s−(m_φ+m_N)²][s−(m_φ−m_N)²]
        kallen = (
            (s - (mphi + mN)**2) * (s - (mphi - mN)**2)
        )
        if kallen <= 0.0:
            return 0.0

        M2_val = self._M2(s, t)
        if M2_val <= 0.0:
            return 0.0

        FF2 = self._helm_F2(-t)   # FF argument is Q² = −t > 0

        dsig = (
            self.Z**2
            / (16.0 * np.pi * kallen)
            * M2_val
            * FF2
        )
        return max(0.0, dsig) * self._GeV2_to_cm2

    # ── PyDarkNewsCrossSection public interface ───────────────────────────────
    # PyDarkNewsCrossSection uses Q2 = −t  (positive convention) internally,
    # so we convert: t = −Q2.

    def diff_xsec_Q2(self, E: float, Q2: float) -> np.ndarray:
        """
        dσ/dQ²  returned as a 0-d numpy array.
        PyDarkNewsCrossSection calls this at lines 429 and 584.
        Q2 ≡ −t > 0.
        """
        t = -Q2
        return np.array(self._dsigma_dt(E, t))

    def total_xsec(self, E: float) -> float:
        """
        σ(E) [cm²] integrated over the physical Q² = −t range.
        """
        mN   = self.MA
        mphi = self.m_phi
        s    = mphi**2 + mN**2 + 2.0 * mN * E

        # Physical t range for φ N → γ N:
        # t_min (most negative) corresponds to backward scattering,
        # t_max (least negative) to forward scattering.
        # For m_γ = 0:
        #   t_max = mphi² − 2E_φ * E_γ_max  ... use standard 2→2 formula
        # Simpler: use Q2 bounds from SIREN's phase_space helpers treating
        # m_ups = 0 (photon).
        Q2min = phase_space.upscattering_Q2min(E, 0.0, self.MA)
        Q2max = phase_space.upscattering_Q2max(E, 0.0, self.MA)
        if Q2max <= Q2min:
            return 0.0
        result, _ = _scipy_integrate.quad(
            lambda Q2: float(np.real(self.diff_xsec_Q2(E, Q2))),
            Q2min,
            Q2max,
            limit=80,
            epsrel=1e-4,
        )
        return max(0.0, result)

    def __repr__(self):
        return (
            f"DarkPrimakoffUpsCase("
            f"m_phi={self.m_phi*1e3:.1f} MeV, "
            f"m_Z'={self.m_Zprime*1e3:.1f} MeV, "
            f"g_n={self.g_n:.2e}, λ={self.lam:.2e} GeV⁻¹)"
        )



import scipy.integrate as _sci_int
import scipy.special  as _sci_sp

# Physical constants (SI-free, natural units with ℏc = 0.197327 GeV·fm)
_M_PI   = 0.13957   # GeV  charged pion mass
_M_MU   = 0.10566   # GeV  muon mass
_M_NU   = 0.0       # GeV  neutrino mass (massless)
_GF     = 1.1664e-5 # GeV⁻²  Fermi constant
_Vud    = 0.9740    # CKM element
_fpi    = 0.1307    # GeV  pion decay constant
_HBAR_C = 0.197327e-13  # GeV·cm  (converts GeV⁻¹ to cm)


def _three_body_decay_rate_phi(m_phi: float, g_mu: float,
                                E_phi_vals: np.ndarray) -> np.ndarray:
 
    M    = _M_PI
    mm   = _M_MU
    mphi = m_phi
    rmu2 = (mm  / M) ** 2
    rph2 = (mphi / M) ** 2

    E_phi_max = (M**2 + mphi**2 - mm**2) / (2.0 * M)
    results   = np.zeros(len(E_phi_vals), dtype=float)

    N_EMU = 120   # y-grid points for trapz integration — pure numpy

    for i, Ep in enumerate(E_phi_vals):
        if Ep <= mphi or Ep >= E_phi_max:
            continue
        x  = Ep / M
        Eq = M - Ep
        q2 = Eq**2 - max(Ep**2 - mphi**2, 0.0)
        if q2 <= mm**2:
            continue
        mq = np.sqrt(q2)
        pq = np.sqrt(max(Eq**2 - q2, 0.0))

        # E_μ bounds (lab = pion rest frame)
        centre  = (q2 + mm**2) / (2.0 * mq)
        half    = pq * (mq**2 - mm**2) / (2.0 * q2) if pq > 0 else 0.0
        Emu_lo  = max(centre - half, mm)
        Emu_hi  = centre + half
        if Emu_hi <= Emu_lo:
            continue

        # Dense y-grid (pure numpy)
        Emu  = np.linspace(Emu_lo, Emu_hi, N_EMU)
        Enu  = M - Ep - Emu                          # E_ν array
        # Mask unphysical points
        good = Enu > 0
        if not good.any():
            continue
        Emu  = Emu[good];  Enu = Enu[good]

        # Four-dot products (all arrays)
        mu_phi = 0.5 * (M**2 - 2.0*M*Enu - mm**2 - mphi**2)
        nu_phi = 0.5 * (M**2 - 2.0*M*Emu - mphi**2)
        pi_nu  = M * Enu
        pi_mu  = M * Emu

        amp2 = 2.0 * (pi_nu * (mm**2 - mu_phi) + pi_mu * nu_phi)
        amp2 = np.clip(amp2, 0.0, None)

        # Trapz integration over E_μ
        results[i] = np.trapz(amp2, Emu)

    prefactor = g_mu**2 * _Vud**2 * _GF**2 * _fpi**2 / (128.0 * np.pi**3)
    return prefactor * results


def build_phi_flux(
    m_phi: float,
    g_mu: float,
    E_thresh: float,
    E_max: float = 3.0,
    n_bins: int  = 50,
    physically_normalized: bool = True,
    dat_dir: str = None,
):

    import siren

    # ── Locate and load the pion dat file ────────────────────────────────────
    if dat_dir is None:
        # siren.utilities.get_resource_path does not exist in this version.
        # Resolve the dat file location the same way the main simulation script
        # does: find where the siren package itself is installed, then walk to
        # the SIREN resources tree which sits two levels above site-packages/siren.
        import siren as _siren_pkg
        _siren_dir  = os.path.dirname(_siren_pkg.__file__)   # .../site-packages/siren
        _site_pkgs  = os.path.dirname(_siren_dir)            # .../site-packages
        # The SIREN source tree is the git clone, found via the running script's
        # location or via a known relative path from the venv.
        # Prefer: walk up from __file__ of this module (SIREN_DarkNews.py) to find
        # the resources/fluxes tree in the SIREN git clone.
        # Fallback chain:
        #   1. ~/SIREN/resources/fluxes/PionKaon/PionKaon-v1.0/
        #   2. next to this file (in case dat files were copied here)
        import pathlib as _pl
        _candidates = [
            _pl.Path.home() / "SIREN" / "resources" / "fluxes" / "PionKaon" / "PionKaon-v1.0",
            _pl.Path(__file__).parent.parent.parent.parent.parent / "SIREN" / "resources" / "fluxes" / "PionKaon" / "PionKaon-v1.0",
            _pl.Path(__file__).parent,
        ]
        dat_dir = None
        for _c in _candidates:
            if (_c / "PionKaon_FHC_pion.dat").exists():
                dat_dir = str(_c)
                break
        if dat_dir is None:
            raise FileNotFoundError(
                "Could not find PionKaon_FHC_pion.dat. "
                "Pass dat_dir= explicitly to build_phi_flux()."
            )

    dat_path  = os.path.join(dat_dir, "PionKaon_FHC_pion.dat")
    all_lines = open(dat_path).readlines()
    headers   = all_lines[0].strip().split()
    data      = [line.strip().split() for line in all_lines[1:] if line.strip()]

    numu_col = headers.index("numu")

    # Pion bin centres [GeV], pion flux [cm⁻² GeV⁻¹ POT⁻¹]
    pi_E   = np.array([(float(r[0]) + float(r[1])) / 2.0 for r in data])
    pi_phi = np.array([float(r[numu_col]) / (50 * 1000 * 1e4) for r in data])
    # (50 bins × 1000 cm² area × 1e4 unit conversion — same as flux.py)

    # ── Output φ energy grid ─────────────────────────────────────────────────
    E_phi_out = np.linspace(E_thresh, E_max, n_bins)
    phi_flux  = np.zeros(n_bins)

    # Rest-frame φ energy grid (fine sampling for the boost integral)
    E_phi_rf_max = (_M_PI**2 + m_phi**2 - _M_MU**2) / (2.0 * _M_PI)
    if E_phi_rf_max <= m_phi:
        # Kinematically forbidden even at rest → return zero flux
        energies = list(E_phi_out)
        flux_arr = list(phi_flux)
        return siren.distributions.TabulatedFluxDistribution(
            E_thresh, E_max, energies, flux_arr, physically_normalized
        )

    E_phi_rf = np.linspace(m_phi, E_phi_rf_max, 200)
    dGamma   = _three_body_decay_rate_phi(m_phi, g_mu, E_phi_rf)

    # ── Fold pion flux × boosted φ spectrum ──────────────────────────────────
    for E_pi, phi_pi in zip(pi_E, pi_phi):
        if phi_pi <= 0.0 or E_pi < _M_PI:
            continue
        # Pion boost: β = p_π / E_π,  γ = E_π / m_π
        p_pi  = np.sqrt(max(E_pi**2 - _M_PI**2, 0.0))
        gamma = E_pi / _M_PI
        beta  = p_pi / E_pi if E_pi > 0 else 0.0

        # For each rest-frame φ energy, sample isotropic decay:
        # lab energy E_lab = γ(E_rf + β p_rf cosθ)
        # Average over cosθ ∈ [−1,1]:  <E_lab> = γ E_rf  (isotropic)
        # But the distribution dN/dE_lab is flat in [γ(E_rf−β p_rf), γ(E_rf+β p_rf)]
        # → project each (E_phi_rf, dΓ/dE_phi_rf) bin onto the lab-frame grid.
        for j, (Erf, dG) in enumerate(zip(E_phi_rf, dGamma)):
            if dG <= 0.0:
                continue
            p_rf   = np.sqrt(max(Erf**2 - m_phi**2, 0.0))
            E_lo   = gamma * (Erf - beta * p_rf)
            E_hi   = gamma * (Erf + beta * p_rf)
            if E_hi <= E_thresh or E_lo >= E_max:
                continue
            if E_lo >= E_hi:
                E_lo = E_hi = gamma * Erf

            # Width of the rest-frame bin
            dErf = (E_phi_rf_max - m_phi) / 199.0

            # Distribute dG * dErf uniformly over [E_lo, E_hi] on the output grid
            # dN/dE_lab = dG * dErf / (E_hi - E_lo)  if E_hi > E_lo
            dE_lab = max(E_hi - E_lo, 1e-9)
            density = dG * dErf / dE_lab * phi_pi   # [GeV⁻¹ cm⁻² POT⁻¹]

            for k, Eout in enumerate(E_phi_out):
                dEout = (E_max - E_thresh) / (n_bins - 1)
                if E_lo <= Eout + 0.5*dEout and Eout - 0.5*dEout <= E_hi:
                    overlap = min(Eout + 0.5*dEout, E_hi) - max(Eout - 0.5*dEout, E_lo)
                    phi_flux[k] += density * overlap

    return siren.distributions.TabulatedFluxDistribution(
        E_thresh, E_max, list(E_phi_out), list(phi_flux), physically_normalized
    )
