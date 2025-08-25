Otimização de Winglet para Aeronaves ✈️
Este projeto implementa um algoritmo de otimização para o design de winglets de aeronaves, visando melhorar o desempenho aerodinâmico. Ele utiliza o otimizador RNSGA-II da biblioteca pymoo para explorar diferentes geometrias de winglets.

Para as análises aerodinâmicas, o código se integra com o programa externo AVL (Athena Vortex Lattice), uma ferramenta comum em análise de desempenho de aeronaves.

Dependências de Software (Windows)
Este projeto depende do programa AVL (Athena Vortex Lattice) para simulações aerodinâmicas no Windows.

Baixe o AVL: Obtenha o executável do AVL para Windows no site oficial do AVL.

Localize o Executável: Após o download, coloque o arquivo avl.exe na mesma pasta onde está o script Python deste projeto.

Requisitos do Python
O script Python utiliza algumas bibliotecas que não são padrão. Para garantir que tudo funcione corretamente, siga estes passos para instalar os requisitos usando o arquivo requirements.txt (que você já possui).

Instale as Bibliotecas Necessárias:
Navegue até o diretório do projeto no seu terminal e instale as bibliotecas usando pip:

pip install -r requirements.txt

Seja feliz para otimizar. :)

obs.: Se for utilizar algum dos perfis da pasta perfis, tem que colocar o que for usar na pasta do executável do AVL.