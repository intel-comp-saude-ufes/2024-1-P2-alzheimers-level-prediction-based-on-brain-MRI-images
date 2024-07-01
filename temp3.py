import os
import shutil

# Caminho do diretório principal que contém as subpastas de níveis de Alzheimer
diretorio_principal = '/home/caio/UFES/Engenharia da Computação/7º Período/TIC/Projeto 2/datasets/KUSHAGRA-FULL-DIVIDED-RIGTH-FILES/train2'
# Caminho do diretório onde todas as imagens serão reunidas
diretorio_destino = '/home/caio/UFES/Engenharia da Computação/7º Período/TIC/Projeto 2/datasets/KUSHAGRA-FULL-DIVIDED-RIGTH-FILES/train/very-mild-demented'

# Percorrer todas as subpastas dentro do diretório principal
for nivel in os.listdir(diretorio_principal):
    caminho_nivel = os.path.join(diretorio_principal, nivel)
    if os.path.isdir(caminho_nivel):
        # Percorrer todas as subpastas dentro da subpasta do nível
        for ressonancia in os.listdir(caminho_nivel):
            caminho_ressonancia = os.path.join(caminho_nivel, ressonancia)
            if os.path.isdir(caminho_ressonancia):
                # Listar todas as imagens na subpasta de ressonância
                for imagem in os.listdir(caminho_ressonancia):
                    caminho_imagem = os.path.join(caminho_ressonancia, imagem)
                    if os.path.isfile(caminho_imagem) and imagem.endswith('.jpg'):
                        # Mover a imagem para o diretório de destino
                        shutil.move(caminho_imagem, os.path.join(diretorio_destino, imagem))
                        print(f'Movido: {imagem} para {diretorio_destino}')

                # Após mover todas as imagens, remover a pasta vazia de ressonância
                os.rmdir(caminho_ressonancia)
        # Opcionalmente, você pode remover a subpasta do nível se ela estiver vazia
        if not os.listdir(caminho_nivel):
            os.rmdir(caminho_nivel)

print('Reorganização concluída com sucesso!')
