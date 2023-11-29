import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import glob
import os

# Função para extrair características usando Hu Moments
def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

# Carregar as imagens e extrair características
def load_data_and_extract_features(data_path):
    images = []
    labels = []

    # Usar a biblioteca glob para buscar arquivos de imagem
    image_files = glob.glob(os.path.join(data_path, "*.png")) + glob.glob(os.path.join(data_path, "*.jpg")) + glob.glob(os.path.join(data_path, "*.jpeg"))

    for image_file in image_files:
        # Obter rótulo do nome do arquivo (assumindo que o rótulo está no nome do arquivo)
        label = os.path.splitext(os.path.basename(image_file))[0]
        image = cv2.imread(image_file)

        # Certifique-se de que a leitura da imagem foi bem-sucedida
        if image is not None:
            features = extract_hu_moments(image)
            images.append(features)
            labels.append(label)

    return np.array(images), np.array(labels)

# Caminho para o conjunto de dados
data_path_covid = r"C:\Processamento-Imagem\img\covid"
data_path_normal = r"C:\Processamento-Imagem\img\normal"

# Carregar dados e extrair características
features_covid, labels_covid = load_data_and_extract_features(data_path_covid)
features_normal, labels_normal = load_data_and_extract_features(data_path_normal)

# Juntar os dados
features = np.concatenate((features_covid, features_normal), axis=0)
labels = np.concatenate((labels_covid, labels_normal), axis=0)

# Dividir os dados em treino e teste
if len(set(labels)) < 2:
    print("Não há classes suficientes para realizar a divisão. Adicione mais dados.")
else:
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Criar e treinar um classificador SVM
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    predictions = classifier.predict(X_test)

    # Avaliar o desempenho do classificador
    accuracy = accuracy_score(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)

    # Imprimir resultados
    print(f'Acurácia: {accuracy}')
    print('Matriz de Confusão:')
    print(confusion_mat)
