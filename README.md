# InScreening-IA

## Integrantes
Leonardo Bruno de Sousa - RM552408
João Vito Santiago da Silva - RM86293
Marco Antônio de Araújo - RM550128
Vinicius Andrade Lopes - RM99343 
Renan Vieira de Jesus - RM551813

## Modelo de Classificação de Imagens

Utilizando técnicas de deep learning, geramos um modelo capaz de distinguir entre pulmões normais e pulmões com câncer, com base em padrões identificados nas imagens de treinamento. O modelo foi treinado utilizando 1000 imagens de cada classe, com 200 épocas, aprendendo a mapear entradas de imagem para suas respectivas classes através de técnicas de aprendizado supervisionado. O modelo foi desenvolvido e treinado utilizando PyTorch.


## Ferramentas e bibliotecas

•	Flask
Essencial para criar a API que receberá as imagens dos pacientes e fornecerá respostas com base na classificação do modelo.

Utilização: Flask cria a estrutura da API, define as rotas (como / e /predict) e manipula as requisições HTTP. Ele também renderiza a página inicial (index.html) e processa as imagens enviadas através do método POST na rota /predict.

•	PyTorch
Usado para desenvolver e executar nosso modelo de classificação de imagens.

Utilização: O modelo pré-treinado é carregado usando torch.load. Este modelo é então usado para prever a classe da imagem fornecida. PyTorch oferece uma API flexível e intuitiva para redes neurais, facilitando o desenvolvimento e a depuração.

•	PIL (Pillow)
Empregado na manipulação de imagens, especialmente para redimensionar e preparar as imagens recebidas dos pacientes antes de passá-las para o modelo de classificação.

Utilização: A imagem enviada é aberta e convertida para RGB. A imagem é então redimensionada para o tamanho esperado pelo modelo (224x224 pixels) e convertida em um tensor.

•	NumPy
Essencial para manipulação de arrays e operações matemáticas, permitindo-nos processar os dados de entrada do modelo de forma eficiente.

Utilização: NumPy é usado para criar arrays do tamanho correto para a entrada do modelo e manipular os resultados da previsão.

•	Jsonify
Usado para converter os resultados das previsões do modelo em formato JSON antes de enviá-los como resposta da API, facilitando a integração com outros sistemas.

Utilização: Os resultados da previsão (classe e confiança) são convertidos em um formato JSON para serem enviados de volta ao cliente que fez a solicitação.

•	DateTime
Utilizado para registrar a data e hora do exame.

Utilização: A data e hora atuais são capturadas no momento da previsão e incluídas na resposta JSON, permitindo que os resultados sejam registrados com um timestamp preciso.

## Apresentação da nossa evolução 
[Vídeo]()

## Links Úteis
Notebook do Modelo: https://colab.research.google.com/drive/1Yx_z55cPYoq9TpsIYg42RGZYbHALiwCN?usp=sharing
Dataset de Imagens de CT de Tórax: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
