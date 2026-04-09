# FMM-AutoOpt
Automated benchmarking for GROMACS-FMM.

## Dependencies
- Python3
- GROMACS with FMM support (Download from: https://www.mpinat.mpg.de/grubmueller/gromacs-fmm-constantph)

## Usage Examples
### Basic Use (Gas-Phase Proteins)
The default parameters are optimized for gas-phase protein simulations.
```bash
python FMM-AutoOpt.py -i protein.gro --top topol.top
```
### Rapid Evaluation
To get results faster, you can decrease the simulation length and limit the range of multipole orders. This is useful for obtaining general performance trends quickly.
```bash
python FMM-AutoOpt.py -i protein.gro --top topol.top -d 2 3 4 -p 0:10:2 0:10:2 0:10:2 --nsteps 200 --resetstep 100 --maxh 0.05
```
### Periodic/Condensed-Phase Systems 
For systems in bulk water or those requiring PBCs in the Coulomb force calculation, provide a custom .mdp file, disable the open boundary default, and use a high-accuracy analytical solution as reference.
```bash
python FMM-AutoOpt.py -i protein.gro --top topol.top --mdp solution.mdp --openboundary 0 --ref 0 10 
```
### Evaluating Sparse Optimization
For a comprehensive benchmark of the FMM_SPARSE criterion, run the pipeline with the flag enabled and disabled.
```bash
python FMM-AutoOpt.py -i protein.gro --top topol.top --sparse 0 -o output_sparse0
python FMM-AutoOpt.py -i protein.gro --top topol.top --sparse 1 -o output_sparse1
```
### Resuming a Run
To continue an abrupted run, supply only the path to the checkpoint file.
```bash
python FMM-AutoOpt.py --cpi checkpoint.json
```
