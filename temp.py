import os
import shutil

# Definir os caminhos do diretório de origem e do diretório de destino
origem = '/home/caio/UFES/Engenharia da Computação/7º Período/TIC/Projeto 2/datasets/KUSHAGRA-FULL/test/non-demented'
destino = '/home/caio/UFES/Engenharia da Computação/7º Período/TIC/Projeto 2/datasets/KUSHAGRA-FULL/train/non-demented'

# Extensões de arquivos de imagem comuns
extensoes_imagem = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

# Listar todos os arquivos no diretório de origem
arquivos = os.listdir(origem)

# Mover cada imagem para o diretório de destino
for arquivo in arquivos:
    # Verificar se o arquivo é uma imagem
    if any(arquivo.lower().endswith(extensao) for extensao in extensoes_imagem):
        caminho_origem = os.path.join(origem, arquivo)
        caminho_destino = os.path.join(destino, arquivo)
        shutil.move(caminho_origem, caminho_destino)
        print(f'Movido: {arquivo}')

print('Todas as imagens foram movidas com sucesso!')
