import numpy as np
import os
import subprocess
import re
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.optimize import minimize

def AVL(Altura, Enflechamento, Cordap, Envergadura, perfil_winglet='EPPLERE850.dat'):
      

    # Arquivo de geometria AVL para uma asa com winglet  
    AVL_Geom = [
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
    f" {4+Enflechamento}           {7+Envergadura}     {Altura}    {Cordap}  0.000\t \n",
    "AFILE \n",
    f"{perfil_winglet}\n",
    "#-----------------------------\n",
    "#==============================================================\n",
    ]

    # linha para escrita de um arquivo de input (fica a critério do usuário colocar mais de uma condição)
    with open('input_geom.avl', 'w') as file:
        file.writelines(AVL_Geom)
       
    input_file_path  = "input_file.txt"
    output_file_path = "output_file.txt"

    # Evitando os erros do avl de timeout e/ou linha de comando dando bug
    try:

        with open(input_file_path, 'w') as input_file:
            input_file.write("load\n")
            input_file.write("input_geom.avl\n")
            input_file.write("case input_run.run\n")
            input_file.write("oper\n")
            input_file.write("x\n")
            input_file.write("fn\n")
            input_file.write(output_file_path + "\n")
            input_file.write("o\n")

        with open(input_file_path, 'r') as input_file:
            subprocess.run(["avl.exe"], stdin=input_file, timeout=25, check=True)

    
    # Evitando de novo erros de executar o avl 
    except FileNotFoundError:
        print("Arquivo AVL não encontrado.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o subprocesso: {e}")
    except subprocess.TimeoutExpired:
        print("Tempo limite de execução excedido.")
    except Exception as e:
        print(f"Erro: {e}")
    finally:

        if os.path.exists(input_file_path):
            os.remove(input_file_path)

   
    with open("output_file.txt", 'r') as file:
        data = file.read()

    # bruxaria da biblioteca re para extrair exatamente os dados que eu quero do arquivo de output
    # Essa expressão vai procurar meus dados de acordo com o padrão que eu defini
    # \d+ para números inteiros, \d+\.\d+ para números decimais, \S+ para strings (sem espaços)
    # A ordem dos parênteses define a ordem dos grupos capturados
    # resumindo: essa coisa feia vai identificar no arquivo de output o padrão da linha que eu quero 
    # A linha que vai ter o valor do meu cl, cd e etc...    

    pattern = r"(\d+)\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+\S+\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+"
    matches = re.findall(pattern, data)

    # Guardando esses dados que eu separei 
    dados = []
    dados = [{
        'CL': float(match[1]),
        'CD': float(match[2]),
        'Cn': float(match[3]),
        'Cl': float(match[4])
    } for match in matches]

    CL, CD, Cn, Cl = dados[0].values()
    return CL, CD, Cn, Cl

# Teste inicial da função AVL pra ver se tá funcionando mesmo
CL, CD, Cn, Cl = AVL(0.5, 0.4, 0.35, 0.55)

# configuração do problema de otimização
class WingletOptimization(Problem):
    def __init__(self):
        # define o problema com 4 variáveis de decisão, 1 objetivo e sem restrições e os limites 
        super().__init__(n_var=4, n_obj=1, n_constr=0, xl=np.array([0.25, 0.0, 0.25, 0.25]), xu=np.array([1.0, 0.5, 0.8, 1.0]))
    
    def _evaluate(self, X, out, *args, **kwargs):

        n = X.shape[0]  # Número de indivíduos na população
        F = np.empty((n, self.n_obj))  # Matriz para armazenar os valores dos objetivos

        for i in range(n):
            # Chama a função AVL para cada indivíduo i na população
            CD, CL, Cn, Cl = AVL(X[i, 0], X[i, 1], X[i, 2], X[i, 3])
            
            # aqui eu digo que meu objetivo é minimizar CD e maximizar CL e Cn
            F[i, 0] = (CD/CL) * Cn

        out["F"] = F


# Exemplo de uso
# Atentar ao tamanho da população e número de gerações para evitar demora excessiva

problem = WingletOptimization()
ref_points = np.array([[0.65, 0.25, 0.35, 0.65], [0.5, 0.3, 0.4, 0.55]])
algorithm = RNSGA2(
    pop_size=100,
    ref_points=ref_points,
    epsilon=0.01,
    normalization='front',
    weights = np.array([[0.5, 0.5, 0.5, 0.5]])
)

# Otimização 
# atentar o número de gerações para evitar demora excessiva
res = minimize(problem,
               algorithm,
               save_history=True,
               termination=('n_gen',10),
               seed=1,
               disp=True,
               verbose = True)

design_values = res.X
objective_values = res.F
historico = res.history

# Plots pessoais de meu interesse
plt.figure()
for gen in historico:
    pop = gen.pop
    design_values       = pop.get("X")  # Acessa as variáveis de projeto (X)
    objective_values    = pop.get("F")  # Acessa os valores dos objetivos (F)
    k                   = objective_values[:, 0]  # Use a primeira coluna de objective_values
    altura              = design_values[:, 0]

    plt.scatter(altura, k, marker='x', alpha=0.5)

plt.xlabel('ALTURA')
plt.ylabel('k')
plt.legend()
plt.show()

plt.figure()
for gen in historico:
    pop = gen.pop
    design_values       = pop.get("X")  # Acessa as variáveis de projeto (X)
    objective_values    = pop.get("F")  # Acessa os valores dos objetivos (F)
    k                   = objective_values[:, 0]  # Use a primeira coluna de objective_values
    enflechamento       = design_values[:, 1]


    plt.scatter(enflechamento, k, marker='+', alpha=0.5)
plt.xlabel('ENFLECHAMENTO')
plt.ylabel('k')
plt.legend()
plt.show()

plt.figure()
for gen in historico:
    pop = gen.pop
    design_values       = pop.get("X")  # Acessa as variáveis de projeto (X)
    objective_values    = pop.get("F")  # Acessa os valores dos objetivos (F)
    k                   = objective_values[:, 0]  # Use a primeira coluna de objective_values
    cordap              = design_values[:, 2]
    plt.scatter(cordap, k, marker='+', alpha=0.5)
plt.xlabel('CORDA NA PONTA')
plt.ylabel('k')
plt.legend()
plt.show()

plt.figure()
for gen in historico:
    pop = gen.pop
    design_values       = pop.get("X")  # Acessa as variáveis de projeto (X)
    objective_values    = pop.get("F")  # Acessa os valores dos objetivos (F)
    k                   = objective_values[:, 0]  # Use a primeira coluna de objective_values
    envergadura         = design_values[:, 3]
    plt.scatter(envergadura, k ,marker='*', alpha=0.5)
plt.xlabel('ENVERGADURA')
plt.ylabel('k')
plt.legend()
plt.show()

