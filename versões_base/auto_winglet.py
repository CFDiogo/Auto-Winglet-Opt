import numpy as np
import os
import subprocess
import re
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Parâmetros de geom para colocar no algoritmo de otimizar
# Tenho 3 funções mas os inputs entram na primeira e os outputs saem na ultima, foda..

def geom(alt, enf, cordp, env):
    
    perfis = ['EPPLERE850.dat', 'NLR7223.dat', 'KC135.dat', 'NLR7223.dat', 'SC20606.dat', 'AH21.dat']
    #perfil_winglet = perfis[perf]
    perfil_winglet = 'NLR7223.dat'
    lines = [
    "jet wing\n",
    "0.8\t              !   Mach\n",
    "0     0     0.0       !   iYsym  iZsym  Zsym\n",
    "9.375  1.5 15.0      !   Sref   Cref   Bref   reference area, chord, span\n",
    "0.25 0.0   0.0        !   Xref   Yref   Zref   moment reference location (arb.)\n",
    "0.01048               !   CDp\n",
    "#\n",
    "#==============================================================\n",
    "#\n",
    "SURFACE\n",
    "Wing\n",
    "20 1.0   20  -2.0  ! Nchord   Cspace   Nspan  Sspace\n",
    "#\n",
    "YDUPLICATE\n",
    "0.0\n",
    "#\n",
    "# twist angle bias for whole surface\n",
    "ANGLE\n",
    "     1.00000    \n",
    "#\n",
    "SCALE\n",
    "  1.0   1.0   1.0\n",
    "#\n",
    "# x,y,z bias for whole surface\n",
    "TRANSLATE\n",
    "    3.  0.  1.2 \n",
    "#\n",
    "#-----------------------------\n",
    "SECTION\n",
    "     0.0          0.0        0.0         2.       0.000 \n",
    "AFILE \n",
    "B737.dat\n",
    "#-----------------------------\n",
    "SECTION\n",
    "     1.0          2.0        0.0         1.35        0.000 \n",
    "AFILE \n",
    "B737.dat\n",
    "#-----------------------------\n",
    "SECTION\n",
    "    2.5           5.0        0.0          1.       0.000  \n",
    "AFILE \n",
    "B737.dat\n",
    "#-----------------------------\n",
    "SECTION  \n",
    "    3.5           7.25        0.0          0.8        0.000  \n",
    "AFILE \n",
    f"{perfil_winglet}\n",
    "#-----------------------------\n",
    "SECTION    ! winglet\n",
    f" {4+enf}           {7+env}     {alt}    {cordp}  0.000\t \n",
    "AFILE \n",
    f"{perfil_winglet}\n",
    "#-----------------------------\n",
    "#==============================================================\n",
    ]

    # Escrever as linhas no arquivo
    with open('input_geom.avl', 'w') as file:
        file.writelines(lines)

               

def exec():
    input_file_path  = "input_file.txt"
    output_file_path = "output_file.txt"

    try:

        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        with open(input_file_path, 'w') as input_file:
            input_file.write("load\n")
            input_file.write("input_geom.avl\n")
            input_file.write("case input_run.run\n")
            input_file.write("oper\n")
            input_file.write("x\n")
            input_file.write("fn\n")
            input_file.write(output_file_path + "\n")
            input_file.write("o\n")

        # Executa o AVL
        with open(input_file_path, 'r') as input_file:
            subprocess.run(["avl.exe"], stdin=input_file, timeout=25, check=True)

    except FileNotFoundError:
        print("Arquivo AVL não encontrado.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o subprocesso: {e}")
    except subprocess.TimeoutExpired:
        print("Tempo limite de execução excedido.")
    except Exception as e:
        print(f"Erro inesperado: {e}")
    finally:

        if os.path.exists(input_file_path):
            os.remove(input_file_path)

def tratar(filename):
    with open(filename, 'r') as file:
        data = file.read()

    pattern = r"(\d+)\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+\S+\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+"
    matches = re.findall(pattern, data)

    dados = []
    for match in matches:
        dados.append({
            'CL': float(match[1]),
            'CD': float(match[2]),
            'Cn': float(match[3]),
            'Cl': float(match[4])
        })
    
    return dados

geom(0.5, 0.4, 0.35, 0.55)  # Altura, Enflechamento, Corda na ponta, Envergadura, Perfil
exec()
dados = tratar("output_file.txt")
output = dados[0]
CL = output['CL']
CD = output['CD']
Cn = output['Cn']
Cl = output['Cl']


class WingletOptimization(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=4, n_constr=0, xl=np.array([0.25, 0.0, 0.25, 0.25]), xu=np.array([1.0, 0.5, 0.8, 1.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Chama a função geom com os parâmetros de entrada x

        CD, CL, Cn, Cl = geom(x[0], x[1], x[2], x[3])  # Salt, enf, cordp, env, perf

        out["F"] = np.array([CD, CL, Cn, Cl])

# Exemplo de uso
problem = WingletOptimization()
ref_points = np.array([[0.25, 0.25], [0.5, 0.5]])
algorithm = RNSGA2(
    pop_size=50,
    ref_points=ref_points,
    epsilon=0.01,
    normalization='front',
    extreme_points_as_reference_points=False,
    weights=np.array([0.5, 0.5])
)


res = minimize(problem,
               algorithm,
               save_history=True,
               termination=('n_gen', 250),
               seed=1,
               disp=False)

Scatter().add(res.F, label="F").show()

