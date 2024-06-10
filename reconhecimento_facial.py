# Libs
from PIL import Image
import numpy as np
import os
import cv2

# Func
def dados_imagem():
    caminhos = [os.path.join('./content/train', f) for f in os.listdir('./content/train')] # Pega o endereço dos arquivos
    faces = []
    ids = []
    for caminho in caminhos:
        if caminho == './content/train\__init__.py': continue
        imagem = Image.open(caminho).convert('L') # Converte para tons de cinza "L"
        imagem_np = np.array(imagem, 'uint8') # Converte para array numpy
        id = int(os.path.split(caminho)[1].split('.')[0].replace('subject', '')) # Pega o id
        ids.append(id)
        faces.append(imagem_np)
    return np.array(ids), faces

def classificador_exist(caminho):
    if os.path.exists(caminho):
        return True
    else:
        return False

# Cria variavel que vai receber o objeto do classificador
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
# Realizar o treinamento
if not classificador_exist('./content/classificadorLBPH.yml'):
    ids, faces = dados_imagem()
    print("Treinando...")
    reconhecedor.train(faces, ids)
    reconhecedor.write('./content/classificadorLBPH.yml')

reconhecedor.read('./content/classificadorLBPH.yml')

# Teste
caminho_teste = './content/test/subject05.sleepy.gif'
imagem_teste = Image.open(caminho_teste).convert('L')
imagem_teste_np = np.array(imagem_teste, 'uint8')
#print(imagem_teste_np)
id_previsto, _ = reconhecedor.predict(imagem_teste_np)
id_correto = int(os.path.split(caminho_teste)[1].split('.')[0].replace('subject', ''))
print(f'id previsto: {id_previsto}, id correto: {id_correto}')

# Visualizando imagem
cv2.putText(imagem_teste_np, 'Previsto: '+str(id_previsto), (5,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.putText(imagem_teste_np, 'Correto: '+str(id_correto), (5,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.imshow('Imagem teste', imagem_teste_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
