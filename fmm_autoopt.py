#!/usr/bin/env python3

import shutil
import argparse
import subprocess
import os
import re
import csv
import sys
import json

GMX_CMD = "gmx"

def check_gromacs(gmx_cmd):
    if not shutil.which(gmx_cmd):
        sys.exit(f"The GROMACS command '{gmx_cmd}' was not found in your PATH.")

class FlushStdout:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def flush(self):
        self.stream.flush()

class CheckpointManager:
    def __init__(self, checkpoint_path):
        self.path = checkpoint_path
        # best_sparse stores '0' or '1' once determined
        self.data = {"accuracy_done": [], "performance_done": [], "best_sparse": None}
        self.run_config = {}

    def load(self):
        """Loads existing checkpoint data."""
        if not os.path.exists(self.path):
            print(f"Error: Checkpoint file {self.path} not found.")
            sys.exit(1)

        with open(self.path, 'r') as f:
            full_checkpoint = json.load(f)

        self.data = full_checkpoint.get("progress", self.data)
        # Backfill missing keys for older checkpoint files.
        self.data.setdefault("accuracy_done", [])
        self.data.setdefault("performance_done", [])
        self.data.setdefault("best_sparse", None)
        self.run_config = full_checkpoint.get("run_config", {})

        print(f"Resuming from checkpoint: {self.path}")
        return self.run_config

    def generate(self, args, mdp=None, param_combos=None, gro=None):
        """Creates the initial checkpoint. Can be called with partial data."""
        self.run_config = self._build_run_config(args, mdp, param_combos, gro)
        self.save_progress()
        print(f"New checkpoint generated at: {self.path}")

    def update_config(self, key, value):
        """Update a specific configuration setting and save immediately."""
        self.run_config[key] = value
        self.save_progress()

    def mark_done(self, stage, item=None):
        """Add an item (like a d,p combo) to the progress list and save."""
        if item:
            if item not in self.data[stage]:
                self.data[stage].append(item)
        else:
            self.data[stage] = True
        self.save_progress()

    def save_progress(self):
        """Writes current state to disk."""
        payload = {"progress": self.data, "run_config": self.run_config}
        with open(self.path, 'w') as f:
            json.dump(payload, f, indent=4)

    def _build_run_config(self, args, mdp, param_combos, gro):
        return {
            "input": os.path.abspath(args.input) if args.input else None,
            "top": os.path.abspath(args.top) if args.top else None,
            "mdp": os.path.abspath(args.mdp) if args.mdp else None,
            "mdp_params": mdp.params if mdp else None,
            "d": args.d,
            "p": args.p,
            "ref": args.ref,
            "param_combos": param_combos,
            "maxerr": args.maxerr,
            "sparse": args.sparse,
            "sparse_test_params": args.sparse_test_params,
            "openboundary": args.openboundary,
            "ntmpi": args.ntmpi,
            "gmx": args.gmx,
            "genvel": args.genvel,
            "maxwarn": args.maxwarn,
            "maxh": args.maxh,
            "nsteps": args.nsteps,
            "resetstep": args.resetstep
        }

    def _print_config_summary(self):
        """Prints summary using current run_config values."""
        c = self.run_config
        mdp_display = c.get('mdp') if c.get('mdp') else "default parameters"

        print("\n" + "="*40)
        print(" CHECKPOINT PARAMETER SUMMARY ".center(40, "="))
        print("="*40)
        print(f"  Input Gro:      {c.get('input')}")
        print(f"  Topology:       {c.get('top')}")
        print(f"  MDP File:       {mdp_display}")
        print(f"  FMM Sparse:     {c.get('sparse')}")
        print(f"  Open Boundary:  {c.get('openboundary')}")
        print(f"  Max Error:      {c.get('maxerr')}")
        
        print("-" * 40)
        print(f"  Max Warnings:   {c.get('maxwarn')}")
        print(f"  Max Wall-time:  {c.get('maxh')} hours")
        print(f"  Steps:          {c.get('nsteps')}")
        print(f"  Clock Reset:    {c.get('resetstep')} steps")
        
        if c.get('param_combos'):
            print(f"  Combos Loaded:  {len(c['param_combos'])} pairs")
        print("="*40 + "\n")


class GroFile:

    def __init__(self, filepath, outdir_root):
        self.filepath = filepath
        self.output_path = outdir_root
        self.num_atoms = 0
        self.atoms = []  # Each atom will be a string line for now
        self.box = None  # Box dimensions (x, y, z)
        self.has_velocities = None

        self._parse()

    def _parse(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        self.num_atoms = int(lines[1].strip())
        self.atoms = lines[2:2 + self.num_atoms]
        
        box_line = lines[2 + self.num_atoms].strip()
        self.box = list(map(float, box_line.split()))[:3] #Note that only first three dimensions of box is checked.

        atom_line = self.atoms[0].rstrip("\n")
        if len(atom_line) > 45:
            self.has_velocities = True
        else:
            self.has_velocities = False

    def print_summary(self):
        print(f"Atoms: {self.num_atoms}")
        print(f"Box dimensions: {self.box}")

    def is_box(self):
        return self.box is not None

    def is_cubic(self):
        x, y, z = self.box
        if x == y == z:
            return True
        else:
            return False

    def change_box(self):
        """ Use gmx editconf to generate a cubic box. """
        outfile_gro = os.path.join(self.output_path, 'cubic_box.gro')
        try:
            subprocess.run([GMX_CMD, "editconf", "-f", self.filepath, "-o", outfile_gro,
                            "-bt", "cubic", "-d", "3"], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("--- GMX EDITCONF FAILURE DETAILS ---")
            print(f"Command failed: {e.cmd}")
            print("\nStandard Output (GROMACS Info):")
            print(e.stdout)
            print("\nStandard Error (GROMACS Fatal Error):")
            print(e.stderr)
            print("----------------------------------")
        self.filepath = outfile_gro
        self._parse()

    def generate_velocities(self, topology, base_mdp, maxwarn, ntmpi):
        output_path = os.path.join(self.output_path, "velocity_generation")
        os.makedirs(output_path, exist_ok=True)

        # 1. Create custom MDP
        mdp_copy = MdpFile()
        mdp_copy.params = base_mdp.params.copy()
        mdp_copy.params['coulombtype'] = 'FMM'
        mdp_copy.params['fmm-override-tree-depth'] = '0'
        mdp_copy.params['fmm-override-multipole-order'] = '0'
        mdp_copy.params['nsteps'] = "1"
        mdp_copy.params['gen-vel'] = "yes"
        mdp_copy.params['continuation'] = 'no'
        mdp_path = os.path.join(output_path, f"generate_velocities.mdp")
        mdp_copy.write(mdp_path)

        # 2. Run simulation
        tpr_path = os.path.join(output_path, f"generate_velocities.tpr")
        tpr = TprFile(mdp_path, self.filepath, topology, tpr_path, maxwarn)
        tpr.generate()

        try:
            subprocess.run([
                GMX_CMD, "mdrun",
                "-deffnm", os.path.join(output_path, f"generate_velocities"),
                "-c", os.path.join(output_path, f"structure_with_velocities.gro"),
                "-ntmpi", str(ntmpi)
            ], check=True, capture_output=True, text=True)
            self.filepath = os.path.join(output_path, f"structure_with_velocities.gro")
            self._parse()
        except subprocess.CalledProcessError as e:
            print("--- GMX MDRUN FAILURE DETAILS ---")
            print(f"Command failed: {e.cmd}")
            print("\nStandard Output (GROMACS Info):")
            print(e.stdout)
            print("\nStandard Error (GROMACS Fatal Error):")
            print(e.stderr)
            print("----------------------------------")
            print("Simulation for generating velocities for accuracy testing failed.")
            raise


class MdpFile:

    DEFAULT_PARAMS = {
        'integrator': 'md',
        'dt': '0.005',
        'comm-mode': 'Angular',
        'nstcomm': '10',
        'verlet-buffer-tolerance':  '-1',
        'rlist': '1',
        'coulombtype': 'FMM',
        'tcoupl': 'v-rescale',
        'tc-grps': 'system',
        'ref-t': '300',
        'tau-t': '0.1',
        'constraints': 'all-bonds',
        'constraint-algorithm': 'Lincs',
        'lincs-order': '6',
        'lincs-iter': '1'
    }

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.params = {}

        if filepath:
            self._read_file(filepath)
        else:
            # Use a copy of default so we can safely modify params later
            self.params = self.DEFAULT_PARAMS.copy()

    def _read_file(self, filepath):
        """Parse an mdp file into a dictionary."""
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):  # Skip empty lines and comments
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    self.params[key.strip()] = value.strip()

    def write(self, outfile_mdp):
        """Write the mdp params back to a file."""
        with open(outfile_mdp, 'w') as f:
            for key, value in self.params.items():
                f.write(f"{key} = {value}\n")



class TprFile:

    def __init__(self, mdp_path, gro_path, top_path, outfile_tpr, maxwarn):
        self.mdp_path = mdp_path
        self.gro_path = gro_path
        self.top_path = top_path
        self.outfile_tpr = outfile_tpr
        self.maxwarn = maxwarn

    def generate(self):
        """Run gmx grompp to create the TPR file."""
        try:
            subprocess.run([
                GMX_CMD, "grompp",
                "-f", self.mdp_path,
                "-c", self.gro_path,
                "-p", self.top_path,
                "-o", self.outfile_tpr,
                "-maxwarn", str(self.maxwarn)
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("--- GMX GROMPP FAILURE DETAILS ---")
            print(f"Command failed: {e.cmd}")
            print("\nStandard Output (GROMACS Info):")
            print(e.stdout)
            print("\nStandard Error (GROMACS Fatal Error):")
            print(e.stderr)
            print("----------------------------------")
            raise

class ParameterSearchSpace:
    def __init__(self, d, p):
        # Validate that lengths match
        if len(d) != len(p):
            raise ValueError(f"d ({len(d)}) and p ({len(p)}) must have the same length.")
        
        self.d = d
        self.p = p

        self.combos = []

        self._generate_combinations()
        self._print_summary()

    def _generate_combinations(self):
        """Parses 'min:max[:step]' strings into (d, p) tuples."""
        for d, p_str in zip(self.d, self.p):
            try:
                parts = p_str.split(':')
                pmin = int(parts[0])
                pmax = int(parts[1])
                step = int(parts[2]) if len(parts) == 3 else 1
                
                if pmin > pmax:
                    print(f"Warning: pmin ({pmin}) > pmax ({pmax}) for d={d}. No combos generated for this d.")
                
                for p in range(pmin, pmax + 1, step):
                    self.combos.append((d, p))
            except (ValueError, IndexError):
                raise ValueError(f"Invalid p_range format: '{p_str}'. Use 'min:max' or 'min:max:step'.")

    def _print_summary(self):
        print("Will test parameter space:")
        # Group by d for a cleaner printout
        summary_dict = {}
        for d, p in self.combos:
            summary_dict.setdefault(d, []).append(p)
        
        for d, p_list in summary_dict.items():
            print(f"  d={d}: p={p_list}")

class SparseTester:

    def __init__(self, grofile, topology, base_mdp, maxh, nsteps, maxwarn, ntmpi, resetstep, outdir_root, test_params):
        self.grofile = grofile
        self.topology = topology
        self.base_mdp = base_mdp
        self.maxh = maxh
        self.nsteps = nsteps
        self.maxwarn = maxwarn
        self.ntmpi = ntmpi
        self.resetstep = resetstep
        self.test_params = test_params

        self.output_path = os.path.join(outdir_root, "sparse_test")
        os.makedirs(self.output_path, exist_ok=True)

        self.best = self.test()

    def test(self):
        
        d, p = self.test_params[0], self.test_params[1]

        # 1. Create custom MDP
        mdp_copy = MdpFile()
        mdp_copy.params = self.base_mdp.params.copy()
        mdp_copy.params['coulombtype'] = 'FMM'
        mdp_copy.params['fmm-override-tree-depth'] = str(d)
        mdp_copy.params['fmm-override-multipole-order'] = str(p)
        mdp_copy.params['nsteps'] = str(self.nsteps)
        if mdp_copy.params.get('tcoupl', '').lower() == 'no':
            print('WARNING! The input .mdp file specifies tcoupl = no, a setting that will not be overriden.\nNot using a thermostat can cause unstable simulations and failure during performance testing.')

        mdp_path = os.path.join(self.output_path, "sparse_test.mdp")
        mdp_copy.write(mdp_path)

        # 2. Generate TPR
        tpr_path = os.path.join(self.output_path, "sparse_test.tpr")
        tpr = TprFile(mdp_path, self.grofile.filepath, self.topology, tpr_path, self.maxwarn)
        tpr.generate()

        # 3. Run simulations
        log_files = {}

        for i in ['0','1']:
            os.environ["FMM_SPARSE"] = i
            tag = f'SPARSE_{i}'
            self._run_simulation(tpr_path, tag)
            log_files[i] = os.path.join(self.output_path, f"{tag}.log")
    
        performance_analysis = PerformanceAnalyzer(log_files)
        performance_analysis.run_all()
        performances = performance_analysis.performances

        for i in performances:
            ns_per_day = performances[i]
            print(f'FMM_SPARSE = {i} ---> {ns_per_day} ns/day')

        best_setting = max(performances, key=performances.get)

        return best_setting

    def _run_simulation(self, tpr_path, tag):
        try:
            subprocess.run([
                GMX_CMD, "mdrun",
                "-deffnm", os.path.join(self.output_path, f"{tag}"),
                "-s", tpr_path,
                "-resetstep", str(self.resetstep),
                "-maxh", str(self.maxh),
                "-ntmpi", str(self.ntmpi)
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("--- GMX MDRUN FAILURE DETAILS ---")
            print(f"Command failed: {e.cmd}")
            print("\nStandard Output (GROMACS Info):")
            print(e.stdout)
            print("\nStandard Error (GROMACS Fatal Error):")
            print(e.stderr)
            print("----------------------------------")
            print("\nSimulation failure during SPARSE testing.")
            raise

class SimulationRunner:

    def __init__(self, grofile, topology, param_space, ref, base_mdp, maxh, nsteps, maxwarn, ntmpi, resetstep, outdir_root, checkpoint=None):
        self.grofile = grofile
        self.topology = topology
        self.param_space = param_space
        self.ref = ref
        self.base_mdp = base_mdp
        self.maxh = maxh
        self.nsteps = nsteps
        self.maxwarn = maxwarn
        self.ntmpi = ntmpi
        self.resetstep = resetstep
        self.outdir_root = outdir_root
        self.checkpoint = checkpoint

        self.edr_files = {}
        self.log_files = {}

    def run_accuracy(self):
        output_path = os.path.join(self.outdir_root, "accuracy_files")
        os.makedirs(output_path, exist_ok=True)
        param_space = self.param_space.copy()

        ref_combo = (self.ref[0], self.ref[1])
        if ref_combo not in param_space:
            param_space.insert(0, ref_combo)

        for combo in param_space:
            d, p = combo[0], combo[1]
            tag = f"d{d}_p{p}"
            edr_file = os.path.join(output_path, f"{tag}.edr")
            if self.checkpoint and tag in self.checkpoint.data.get("accuracy_done", []):
                if os.path.exists(edr_file):
                    self.edr_files[tag] = edr_file
                print(f"Skipping accuracy for {tag} (already in checkpoint).")
                continue

            print(f"Running simulations for accuracy testing with {tag}")

            # 1. Create custom MDP
            mdp_copy = MdpFile()
            mdp_copy.params = self.base_mdp.params.copy()
            mdp_copy.params['coulombtype'] = 'FMM'
            mdp_copy.params['fmm-override-tree-depth'] = str(d)
            mdp_copy.params['fmm-override-multipole-order'] = str(p)
            mdp_copy.params['nsteps'] = "1"
            mdp_copy.params['tcoupl'] = "no"
            mdp_copy.params['pcoupl'] = "no"
            mdp_copy.params['continuation'] = "yes"
            mdp_copy.params['gen-vel'] = "no"
            mdp_copy.params['nstenergy'] = "1"
            mdp_copy.params['nstcalcenergy'] = "1"
            mdp_copy.params['constraints'] = "none"
            mdp_copy.params['comm-mode'] = "none"
            mdp_path = os.path.join(output_path, f"{tag}.mdp")
            mdp_copy.write(mdp_path)

            # 2. Run simulation
            tpr_path = os.path.join(output_path, f"{tag}.tpr")
            tpr = TprFile(mdp_path, self.grofile.filepath, self.topology, tpr_path, self.maxwarn)
            tpr.generate()

            try:
                subprocess.run([
                    GMX_CMD, "mdrun",
                    "-deffnm", os.path.join(output_path, f"{tag}"),
                    "-ntmpi", str(self.ntmpi)
                ], check=True, capture_output=True, text=True)
                self.edr_files[tag] = edr_file
                if self.checkpoint:
                    self.checkpoint.mark_done("accuracy_done", tag)

            except subprocess.CalledProcessError as e:
                print("--- GMX MDRUN FAILURE DETAILS ---")
                print(f"Command failed: {e.cmd}")
                print("\nStandard Output (GROMACS Info):")
                print(e.stdout)
                print("\nStandard Error (GROMACS Fatal Error):")
                print(e.stderr)
                print("----------------------------------")

    def run_performance(self):
        output_path = os.path.join(self.outdir_root, "performance_files")
        os.makedirs(output_path, exist_ok=True)
        for combo in self.param_space:
            d, p = combo[0], combo[1]
            tag = f"d{d}_p{p}"
            log_file = os.path.join(output_path, f"{tag}.log")
            if self.checkpoint and tag in self.checkpoint.data.get("performance_done", []):
                if os.path.exists(log_file):
                    self.log_files[tag] = log_file
                print(f"Skipping performance for {tag} (already in checkpoint).")
                continue

            print(f"Running simulations for performance testing with {tag}")

            # 1. Create custom MDP
            mdp_copy = MdpFile()
            mdp_copy.params = self.base_mdp.params.copy()
            mdp_copy.params['coulombtype'] = 'FMM'
            mdp_copy.params['fmm-override-tree-depth'] = str(d)
            mdp_copy.params['fmm-override-multipole-order'] = str(p)
            mdp_copy.params['nsteps'] = str(self.nsteps)
            if mdp_copy.params.get('tcoupl', '').lower() == 'no':
                print('WARNING! The input .mdp file specifies tcoupl = no, a setting that will not be overriden.\nNot using a thermostat can cause unstable simulations and failure during performance testing.')

            mdp_path = os.path.join(output_path, f"{tag}.mdp")
            mdp_copy.write(mdp_path)

            # 2. Run simulation
            tpr_path = os.path.join(output_path, f"{tag}.tpr")
            tpr = TprFile(mdp_path, self.grofile.filepath, self.topology, tpr_path, self.maxwarn)
            tpr.generate()

            try:
                subprocess.run([
                    GMX_CMD, "mdrun",
                    "-deffnm", os.path.join(output_path, f"{tag}"),
                    "-resetstep", str(self.resetstep),
                    "-maxh", str(self.maxh),
                    "-ntmpi", str(self.ntmpi)
                ], check=True, capture_output=True, text=True)
                self.log_files[tag] = log_file
                if self.checkpoint:
                    self.checkpoint.mark_done("performance_done", tag)

            except subprocess.CalledProcessError as e:
                print("--- GMX MDRUN FAILURE DETAILS ---")
                print(f"Command failed: {e.cmd}")
                print("\nStandard Output (GROMACS Info):")
                print(e.stdout)
                print("\nStandard Error (GROMACS Fatal Error):")
                print(e.stderr)
                print("----------------------------------")


class AccuracyAnalyzer:

    def __init__(self, edr_files, ref, num_atoms, dt, outdir_root):
        self.edr_files = edr_files  # dict[tag] = edr file
        self.ref = ref
        self.num_atoms = num_atoms
        self.energy_terms = ["Coulomb-(SR)", "Coulomb-14"]
        self.output_path = os.path.join(outdir_root, 'accuracy_files')
        self.energy = {}
        self.errors = {}

    def run_all(self):
        self.gmx_energy()
        self.compute_error()

    def _get_xvg_path(self, tag):
        return os.path.join(self.output_path, f"{tag}_CoulE.xvg")

    def gmx_energy(self):
        selection_input = "\n".join(self.energy_terms) + "\n0\n"

        for tag, edr_path in self.edr_files.items():
            output_xvg = self._get_xvg_path(tag)

            if not os.path.exists(edr_path):
                print(f"Skipping {tag}: EDR file missing.")
                continue

            try:
                subprocess.run(
                    [GMX_CMD, 'energy', '-f', edr_path, '-o', output_xvg, 
                    '-b', '0', '-e', '0', '-dp'],
                    input=selection_input, text=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print("--- GMX ENERGY FAILURE DETAILS ---")
                print(f"Command failed: {e.cmd}")
                print("\nStandard Output (GROMACS Info):")
                print(e.stdout)
                print("\nStandard Error (GROMACS Fatal Error):")
                print(e.stderr)
                print("----------------------------------")            
                raise

    def _parse_xvg(self, xvg):
        valid_line = None
        if not os.path.exists(xvg):
            return None

        with open(xvg, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith(('#', '@')):
                    valid_line = line
                    break

        parts = valid_line.strip().split()
        try:
            if len(parts) == 3:
                return sum([float(parts[1]), float(parts[2])])
            elif len(parts) == 2:
                return float(parts[1])
        except (ValueError, IndexError):
            return None

    def compute_error(self):
        for tag in self.edr_files:
            xvg_path = self._get_xvg_path(tag)
            result = self._parse_xvg(xvg_path)
            
            if result is not None:
                self.energy[tag] = result 
            else:
                self.energy[tag] = None
                self.errors[tag] = None
                print(f"Failed to parse {xvg_path} during accuracy analysis.")

        ref_tag = f'd{self.ref[0]}_p{self.ref[1]}'
        if ref_tag not in self.energy or self.energy[ref_tag] is None:
            print(f"Critical Error: Reference '{ref_tag}' failed. Accuracy cannot be calculated.")
            raise

        Eref = self.energy[ref_tag]

        for tag in self.energy:
            if self.energy[tag] == None:
                continue
            
            if tag == ref_tag:
                self.errors[tag] = 0.0
            
            E = self.energy[tag]
            # Compute error and convert from kJ/mol to J/mol/atom
            error = abs(E - Eref) / self.num_atoms * 1000
            self.errors[tag] = error

class PerformanceAnalyzer:

    def __init__(self, log_files):
        self.log_files = log_files
        self.performances = {}

    def run_all(self):
        for tag, log_path in self.log_files.items():
            self.performances[tag] = self._parse_performance(log_path)

    def _parse_performance(self, log_path):
        """
        Extracts the ns/day performance from a GROMACS log file.
        Returns None if the performance line is not found or can't be parsed.
        """
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"File not found: {log_path}")
            return None

        for line in reversed(lines):
            if line.strip().startswith("Performance:"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])  # first value is ns/day
                    except ValueError:
                        print(f"Failed to parse performance from line: {line}")
                        return None

        return None


class SummarizeResults:

    def __init__(self, errors, performances, threshold, outdir_root):
        self.errors = errors
        self.performances = performances
        self.threshold = threshold
        self.output_path = outdir_root

    def print_best_params(self):
        """
        Find the (d,p) combo with highest performance among those
        with error below the given threshold.
        
        Returns: (best_tag, best_perf)
        """
        # 1. Filter accurate parameters
        candidate_tags = set(self.errors.keys()) & set(self.performances.keys())
        valid_and_accurate = []

        for tag in candidate_tags:
            err = self.errors[tag]
            perf = self.performances[tag]

            # Skip if data is missing or if the error exceeds the threshold
            if err is None or perf is None:
                continue
            
            if err < self.threshold:
                valid_and_accurate.append(tag)

        if not valid_and_accurate:
            print(f"No parameter sets provided both valid performance data and error below the defined threshold of {self.threshold} J/mol/K.")
            return

        # 2. Determine the highest performing valid combination
        best_tag = max(valid_and_accurate, key=lambda t: self.performances[t])
        best_perf = self.performances[best_tag]
        best_error = self.errors[best_tag]

        # Formatting the tag for clear output
        clean_tag = best_tag.replace('_', '/')
        
        print(f"Optimal parameters identified: {clean_tag}")
        print(f"Calculated Error: {best_error:.4e} J/mol/atom")
        print(f"Reported Performance: {best_perf:.2f} ns/day")

    def write_summary(self):
        outfile_csv = os.path.join(self.output_path, "summary.csv")

        def sort_key(tag):
            temp_str = tag.lstrip("d")
            d_str, p_str = temp_str.split("_p")
            return (int(d_str), int(p_str))
        
        all_tags = sorted(set(self.errors.keys()) | set(self.performances.keys()), key=sort_key)

        with open(outfile_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([";d", "p", "Error (J/mol/atom)", "Performance (ns/day)", "Meets Threshold?"])

            for tag in all_tags:
                d_str, p_str = tag.lstrip("d").split("_p")
                d, p = int(d_str), int(p_str)

                error_drift = self.errors.get(tag)
                perf = self.performances.get(tag)

                if error_drift is not None and error_drift < self.threshold:
                    accurate = "Yes"
                elif error_drift is not None:
                    accurate = "No"
                else:
                    accurate = "N/A (Failed)"

                error_str = f"{error_drift:.4e}" if error_drift is not None else "N/A"
                perf_str = f"{perf:.2f}" if perf is not None else "N/A"

                writer.writerow([d, p, error_str, perf_str, accurate])

        print(f"Summary of results written to {outfile_csv}")

def prepare_output_dir(path):
    """
    Check if `path` exists. If so, back it up as path_backup1, path_backup2, etc.
    Then create a fresh directory.
    """
    if os.path.exists(path):
        i = 1
        while True:
            backup_path = f"{path}_bck{i}"
            if not os.path.exists(backup_path):
                break
            i += 1
        print(f"Output directory {path} already exists. Backing up to {backup_path}")
        shutil.move(path, backup_path)
    os.makedirs(path, exist_ok=True)
    print(f"Output will be written to directory {path}")
    return


class Formatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

###########################################################################################

parser = argparse.ArgumentParser(description=("     *** FMM-AutoOpt ***\n"
        "Automated benchmarking for GROMACS-FMM.\n"
        "Author: Louise J. Persson (louise.persson@kemi.uu.se)\n"
        "Please cite: XXX\n\n"
        "Usage example:\n"
        "    python fmm-autoopt.py -i system.gro --top system.top --d 2 3 --p 0:20 0:20\n\n"
        "Important note:\n"
        "  The system must be energy-minimized and equilibrated prior to \n"
        "  benchmarking to ensure stable trajectories and accurate timing."),
        formatter_class=Formatter)

# --- Meta ---
parser.add_argument('--version', action='version', version='FMM-AutoOpt 1.0')

# --- Required Inputs ---
req = parser.add_argument_group('required arguments')
req.add_argument("-i", "--input", default=None, help="Input .gro file")
req.add_argument("--top", default=None, help="Topology .top file")

# --- Search Space & Accuracy ---
fmm = parser.add_argument_group('FMM search space & reference')
fmm.add_argument("-d", "--depth", dest="d", nargs='+', type=int, default=[2, 3, 4],
    help="Tree depths to benchmark (e.g., -d 2 3 4).")
fmm.add_argument("-p", "--p-range", dest="p", nargs='+', type=str, default=["0:20:2", "0:20:2", "0:20:2"],
    help="Multipole order ranges for each d (e.g., '0:20:2'). Must match -d length.")
fmm.add_argument("--ref", nargs=2, type=int, default=[0, 0], metavar=('d', 'p'),
    help="Reference d/p for accuracy testing.")
fmm.add_argument("--maxerr", type=float, default=0.01, 
    help="Accuracy threshold (J/mol/atom).")

# --- Performance & Environment ---
perf = parser.add_argument_group('performance & environment')
perf.add_argument("--sparse", choices=["0", "1", "auto"], default="0", 
    help="Set FMM_SPARSE. 'auto' benchmarks --sparse-test-params to decide.")
perf.add_argument("--sparse-test-params", nargs=2, type=int, default=[3, 8], metavar=('d', 'p'),
    help="The (d, p) combo used if --sparse is 'auto'.")
perf.add_argument("--openboundary", choices=["1", "0"], default="1", 
    help="Value for environment variable OPENBOUNDARY.")

# --- GROMACS Execution Control ---
gmx = parser.add_argument_group('GROMACS settings')
gmx.add_argument("--gmx", default="gmx", help="GROMACS executable or path.")
gmx.add_argument("--mdp", help="Optional .mdp file (defaults to vacuum settings).")
gmx.add_argument("--nsteps", type=int, default=1000, help="Steps per simulation.")
gmx.add_argument("--maxh", type=float, default=0.25, help="Max wall clock time (hours).")
gmx.add_argument("--maxwarn", type=int, default=1, help="Maximum warnings to allow for gmx grompp.")
gmx.add_argument("--resetstep", type=int, default=200, help="Number of steps after which the clock is reset for performance testing.")
gmx.add_argument("--ntmpi", default="1", help="Number of thread-MPI ranks.")
gmx.add_argument("--genvel", action="store_true", help="Generate random velocities.")

# --- Misc ---
parser.add_argument("-o", "--out", default="output_FMM-AutoOpt", help="Output directory.")
parser.add_argument("--cpi", help="Checkpoint file to resume a run.")

#########################################################################################

def main():
    
    # Initiate
    args = parser.parse_args()
    sys.stdout = FlushStdout(sys.stdout)
    sys.stderr = FlushStdout(sys.stderr)
    
    # Prepare GROMACS
    global GMX_CMD
    check_gromacs(args.gmx)
    GMX_CMD = args.gmx

    # Checkpoint Initialization
    ckpt_path = args.cpi if args.cpi else os.path.join(args.out, "checkpoint.json")
    checkpoint = CheckpointManager(ckpt_path)

    if args.cpi:
        run_config = checkpoint.load()
        for key, value in run_config.items():
            setattr(args, key, value)
        args.out = os.path.dirname(os.path.abspath(args.cpi))
    else:
        if not args.input or not args.top:
            parser.error("-i/--input and --top are required unless --cpi is provided.")
        prepare_output_dir(args.out)
        checkpoint.generate(args, mdp=None, param_combos=None)

    # System Setup (Files and Environment)
    mdp = MdpFile(args.mdp) if args.mdp else MdpFile()
    gro = GroFile(args.input, args.out)

    if not args.cpi:
        if not gro.is_box() or not gro.is_cubic():
            print('Creating a cubic box...')
            gro.change_box()
            checkpoint.update_config("input", gro.filepath)

        if not gro.has_velocities or args.genvel:
            print("Generating velocities...")
            gro.generate_velocities(args.top, mdp, args.maxwarn, args.ntmpi)
            checkpoint.update_config("input", gro.filepath)
    else:
        print(f"Using processed structure from checkpoint: {gro.filepath}")
        gro.print_summary()

    os.environ["OPENBOUNDARY"] = args.openboundary
    print(f"Using OPENBOUNDARY={os.environ['OPENBOUNDARY']}")

    # Parameter Space setup
    if not args.cpi:
        try:
            param_space = ParameterSearchSpace(args.d, args.p)
            param_combos = param_space.combos
            checkpoint.update_config("param_combos", param_combos)
            checkpoint.update_config("mdp_params", mdp.params)
        except ValueError as e:
            parser.error(str(e))
    else:
        param_combos = args.param_combos
        mdp.params = args.mdp_params

    # Sparse Logic
    if args.sparse == 'auto':
        best_sparse = checkpoint.data.get("best_sparse")
        if not best_sparse:
            if not args.sparse_test_params:
                parser.error("--sparse 'auto' requires --sparse-test-params.")
            print(f'Evaluating optimal FMM_SPARSE using d={args.sparse_test_params[0]}, p={args.sparse_test_params[1]}')
            sparse_test = SparseTester(gro, args.top, mdp, args.maxh, args.nsteps, args.maxwarn, args.ntmpi, args.resetstep, args.out, args.sparse_test_params)
            best_sparse = sparse_test.best
            checkpoint.data["best_sparse"] = best_sparse
            checkpoint.save_progress()

        os.environ["FMM_SPARSE"] = best_sparse
    else:
        os.environ["FMM_SPARSE"] = args.sparse
        print(f"Using FMM_SPARSE={os.environ['FMM_SPARSE']}")

    # Execution & Analysis
    runner = SimulationRunner(gro, args.top, param_combos,args.ref, mdp, args.maxh, args.nsteps, args.maxwarn, args.ntmpi, args.resetstep, args.out, checkpoint)
    runner.run_accuracy()
    runner.run_performance()

    accuracy = AccuracyAnalyzer(runner.edr_files, args.ref, gro.num_atoms, mdp.params['dt'], args.out)
    accuracy.run_all()
    performance = PerformanceAnalyzer(runner.log_files)
    performance.run_all()

    SummarizeResults(accuracy.errors, performance.performances, args.maxerr, args.out).write_summary()


if __name__ == "__main__":
    main()
