# dEFEND-model

Este repositório contém o código do modelo de aprendizado de máquina `dEFEND`, uma ferramenta que utiliza tanto o conteúdo de notícias quanto comentários e postagens de usuários para realizar detecção explicável de fake news. Foi desenvolvido originalmente por pesquisadores da Pennsylvania State University e da Arizona State University. A versão contida aqui foi modificada para atender aos requisitos do [Projeto LIBRA](https://github.com/Projeto-LIBRA).

## Arquivos

- `defend.py`: Contém a construção do modelo, treinamento e pesos de atenção originais do `dEFEND`.
- `go_defend.py`: Arquivo criado com base no `dEFEND` original, contendo a função principal para treinamento do modelo. Envolve o carregamento do dataset de treinamento e o pré-processamento dos dados, incluindo a divisão do teste de treinamento.
- `test_defend.py`: Arquivo criado para executar o modelo treinado resultante do `go_defend.py`, de forma mais leve e rápida. Estruturado para ser executado em uma função Lambda na AWS, lendo arquivos do S3 e gerando uma resposta no Twitter ao final da execução.
- `Dockerfile`: configura o container utilizado para deploy da função Lambda.
- `deploy.sh`: script de deploy da imagem do container da função Lambda para o Amazon ECR.

## Treinamento do modelo

O modelo foi treinado utilizando o dataset [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet), que pode ser obtido executando o código disponível em seu repositório. São necessárias chaves de acesso à API do Twitter para coletar os dados, que podem ser obtidas para propósitos não-comerciais e de pesquisa. Com o dataset coletado, o treinamento pode ser feito executando os seguintes passos.

1. Insira o caminho do dataset e o caminho dos vetores de palavra no arquivo `go_defend.py`
2. Altere o nome do dataset na função `main`: "politifact" ou "gossipcop" para o FakeNewsNet.
3. Execute o arquivo `go_defend.py`. Serão gerados os arquivos de treinamento do modelo.
