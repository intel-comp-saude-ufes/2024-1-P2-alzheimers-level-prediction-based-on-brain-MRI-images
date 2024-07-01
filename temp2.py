import os
import shutil
import re

# Caminho do diretório principal que contém as subpastas de níveis de Alzheimer
diretorio_principal = '/home/caio/UFES/Engenharia da Computação/7º Período/TIC/Projeto 2/datasets/KUSHAGRA-FULL-2/train'

# Função para obter a pasta de destino com base no nome do arquivo
def obter_pasta_destino(nome_arquivo):
    match = re.match(r'OAS1_(\d+)_MR2_mpr-(\d+)_\d+\.jpg', nome_arquivo)
    if match:
        numero1 = match.group(1)
        numero2 = match.group(2)
        return f'{numero1}_MR2_{numero2}'
    return None

# Percorrer todas as subpastas dentro do diretório principal
for nivel in os.listdir(diretorio_principal):
    caminho_nivel = os.path.join(diretorio_principal, nivel)
    if os.path.isdir(caminho_nivel):
        # Listar todas as imagens na subpasta
        for imagem in os.listdir(caminho_nivel):
            caminho_imagem = os.path.join(caminho_nivel, imagem)
            if os.path.isfile(caminho_imagem) and imagem.endswith('.jpg'):
                pasta_destino = obter_pasta_destino(imagem)
                if pasta_destino:
                    caminho_pasta_destino = os.path.join(caminho_nivel, pasta_destino)
                    if not os.path.exists(caminho_pasta_destino):
                        os.makedirs(caminho_pasta_destino)
                    shutil.move(caminho_imagem, os.path.join(caminho_pasta_destino, imagem))
                    print(f'Movido: {imagem} para {caminho_pasta_destino}')

print('Organização concluída com sucesso!')
