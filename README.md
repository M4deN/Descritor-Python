# Projeto de Processamento de Imagens Médicas

Este projeto utiliza técnicas de processamento de imagens para a classificação de imagens médicas relacionadas a COVID-19. A implementação atual utiliza o descritor de Hu Moments para extrair características das imagens e treina um classificador SVM (Support Vector Machine) para realizar a classificação. A acurácia do modelo é avaliada no conjunto de teste, e uma matriz de confusão é gerada para uma análise mais detalhada.

## Requisitos do Projeto

### Execução do Código

Certifique-se de ter os seguintes requisitos instalados no ambiente de execução:

- Python (versão 3.0 ou superior)
- Bibliotecas necessárias: `cv2`, `numpy`, `scikit-learn`

Para instalar as bibliotecas, utilize o seguinte comando:

1. **Clone o Repositório:**
   ```bash
   git clone https://github.com/seu-usuario/Descritor-Python.git
   cd Descritor-Python
   ```

2. **Configuração do Ambiente Virtual (opcional, mas recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # No Windows: venv\Scripts\activate
   ```

3. **Instale as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```
     ```bash
    pip install opencv-python numpy scikit-learn
    ```

4. **Execute o Script:**
   ```bash
   python Descry.py
   ```

### Estrutura do Conjunto de Dados

Organize o conjunto de dados de acordo com a seguinte estrutura:

```
- C:\Processamento-Imagem\img
    - covid
        - imagem1.png
        - imagem2.jpg
        - ...
    - normal
        - imagem3.jpeg
        - imagem4.png
        - ...
```

### Executando o Código

Execute o script `Descry.py` para treinar o modelo, fazer previsões no conjunto de teste e imprimir os resultados. Certifique-se de que o caminho do conjunto de dados (`data_path_covid` e `data_path_normal`) esteja corretamente configurado no script.

```bash
python Descry.py
```

### Resultados Esperados

O script imprimirá a acurácia do modelo no conjunto de teste e a matriz de confusão, fornecendo uma avaliação do desempenho do classificador SVM treinado.


```markdown
# Projeto de Classificação de Imagens de Raio-X (Descritor Hu Moments)

Este projeto visa a classificação de imagens de raio-X em duas categorias: COVID-19 e Normal. A classificação é realizada utilizando o descritor Hu Moments como método de extração de características, e um classificador SVM (Support Vector Machine).

## Estrutura do Projeto

- `Descry.py`: Script principal contendo a implementação do projeto.
- `README.md`: Este arquivo, fornecendo informações detalhadas sobre o projeto.
- `img/`: Diretório contendo subdiretórios `covid` e `normal` com as imagens de treinamento.

## Requisitos do Ambiente

- Python 3.x
- Bibliotecas: cv2, numpy, scikit-learn

Instale as bibliotecas necessárias usando o seguinte comando:

```bash
pip install opencv-python numpy scikit-learn
```

## Resultados

Após a execução do script, os resultados serão impressos no console. A acurácia e a matriz de confusão indicarão o desempenho do modelo SVM na classificação das imagens de raio-X.

## Desenvolvimento Adicional

Se desejar experimentar outros descritores, edite a função `extract_hu_moments` em `Descry.py`. Considere explorar descritores como LBP (Local Binary Pattern) e testar diferentes configurações do SVM para otimizar o desempenho.

## Vídeo de Apresentação

Um vídeo de apresentação do projeto, explicando o desenvolvimento, os resultados e a execução, está disponível [aqui](link-do-video).

---

**Observação:** Este README é um exemplo genérico. Personalize-o conforme necessário para refletir as características específicas do seu projeto.
```