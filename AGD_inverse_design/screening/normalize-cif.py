import os
import random
from pymatgen import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar 
from pymatgen.core.lattice import Lattice
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation
import multiprocessing as mp


root = 'inverse_design_target/'
files = os.listdir(root)
print(len(files))

def function(file):
    crystal = Structure.from_file(root+file)
    t = ConventionalCellTransformation()
    crystal = t.apply_transformation(crystal)

    crystal.to(fmt='cif',\
         filename='inverse_design_target_symgroup/'+file,\
         symprec=0.1)

pool = mp.Pool(processes=8)
pool.map(function, files)
