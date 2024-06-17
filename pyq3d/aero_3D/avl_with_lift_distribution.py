import os
import subprocess as sp
import tempfile
from pathlib import Path

from aerosandbox.aerodynamics.aero_3D.avl import AVL


class AVLWithLiftDistribution(AVL):

    def run_lift_distribution(self, cl=None, alpha=0, verbose=0):

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            if self.working_directory is not None:
                directory = Path(self.working_directory)  # For debugging

            verbosity = {1: None, 0: open(os.devnull, "w")}

            airplane_file = "airplane.avl"
            self.write_avl(directory / airplane_file)

            avl = sp.Popen(
                [self.avl_command],
                stdin=sp.PIPE,
                stdout=verbosity[verbose],
                stderr=None,
                creationflags=sp.CREATE_NEW_PROCESS_GROUP,
                universal_newlines=True,
            )

            avl.stdin.write(f"LOAD {airplane_file}" + "\n")
            avl.stdin.write("OPER" + "\n")
            if cl:
                avl.stdin.write(f"a c {cl}" + "\n")
            else:
                avl.stdin.write(f"a a {alpha}" + "\n")
            avl.stdin.write("M" + "\n")
            avl.stdin.write(f"MN {self.op_point.mach()}" + "\n")
            avl.stdin.write(f"v {self.op_point.velocity}" + "\n")
            avl.stdin.write(f"d {self.op_point.atmosphere.density()}" + "\n")
            avl.stdin.write(f"g 9.81" + "\n")
            avl.stdin.write("\n")
            avl.stdin.write("xx" + "\n")
            avl.stdin.write(f"fe {directory / 'fe.txt'}" + "\n")
            avl.stdin.write("o" + "\n")
            avl.stdin.write(f"ft {directory / 'ft.txt'}" + "\n")
            avl.stdin.write("o" + "\n")
            avl.stdin.write("\n")
            avl.stdin.write("quit" + "\n")
            avl.stdin.close()
            avl.wait()

            dict_fe = {"Yle": [], "cl": [], "c": []}

            with open(directory / "fe.txt") as file:
                lines = file.readlines()

                for l in lines:
                    if "Yle" in l or "cl" in l:
                        l_mod = [s.strip() for s in l.split("=")]
                        values = " ".join(l_mod).replace("  ", " ").split(" ")
                        dict_fe[values[0]].append(float(values[1]))
                    if "Ave. Chord   =" in l:
                        l_mod = [s.strip() for s in l.split("=")]
                        values = " ".join(l_mod).replace("  ", " ").split(" ")
                        dict_fe["c"].append(float(values[5]))

            return dict_fe
