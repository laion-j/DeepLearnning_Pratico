# Documentação Redes Neurais - (básico)

Estas são algumas anotações que fiz para a parte teórica do curso, procurando entender os conceitos básicos de DP.

<br>

# - Perceptron
Perceptron é um classificador linear que se assemelha a um neurônio.

![Perceptron](/Images/Perceptron.png)

*Encontrado em: https://www.embarcados.com.br/wp-content/uploads/2016/09/Perceptron-01.png*

### Composto por (em relação à imagem):
- Entrada: representado por x.
  - Valor de input para a rede.
- Pesos: representados por w.
  - Valores multiplicados pelo input que geram o resultado esperado ou estimado. Gerados "aleatoriamente" ao longo das épocas.
- Bias: representada pelo ômega.
  - Valor que é somado a função, também representado por b. É uma constante que ajuda o modelo a se adpatar melhor aos dados. Translada a função no eixo.
- FUNÇÃO DE SOMA: representada pelo sigma
- Função de ativação: representada por g(.)
  - Explorado na seção
- Saída: representada por y
  - Resultado final.
  
```
(X1 * W1 + X1 * W2 + X1 * W3) +
(X1 * W1 + X1 * W2 + X1 * W3) +
(X1 * W1 + X1 * W2 + X1 * W3) + ÔMEGA = Y
```

>## SIGMA(X * W) + B = Y

>## SOMA(ENTRADAS * PESOS) + BIAS = RESULTADO

<br>
<br>

# Cálculo do erro e Aprendizagem
O cálculo do erro é o que irá determinar a acuracidade da rede neural.

<br>

### Algoritmo mais simples:
>### erro = respostaCorreta - respostaCalculada

### Mean Square Error - (mais usado):
>### É a somatória dos quadrados da expressão de erro simples (acima) dividido pela quantidade de elementos.

### Root Mean Square Error:
>### É a Raiz quadrada da somatória dos quadrados da expressão de erro simples (acima) dividido pela quantidade de elementos.

<br>

### Descida de Gradiente - Gradient Descent
Calcular a derivada parcial para mover a direção do grandiente.

<br>

![DescidaGradiente](/Images/descidaGradiente.jpg)
*Print da tela da aula 109 - Seção 11: Anexo I*

- Estocástica (SGD - Descida do Gradiente Estocástica)
  - Ajuda a previnir mínimos locais.
  - Mais rápido.

<br>

## Taxa de Aprendizagem - Learning Rate

A taxa de aprendizagem refere-se à "velocidade (tamanho do passo)" da descida do gradiente. Assim, quanto menor a taxa, melhor para a descida, evitando ultrapassar o limite e o processo de overshooting.

<br>

Neste capítulo do livro Deep Learning Book explica como escolher a taxa de aprendizagem.
>Para maiores informacões -> [Taxa de Aprendizagem: como definir.](https://www.deeplearningbook.com.br/a-taxa-de-aprendizado-de-uma-rede-neural/)

<br>

## Épocas - Epochs

Número de atualização de pesos.

Neste capítulo do livro Deep Learning Book explica como escolher o número de épocas.
>Para maiores informacões -> [Taxa de Aprendizagem: como definir.](https://www.deeplearningbook.com.br/usando-early-stopping-para-definir-o-numero-de-epocas-de-treinamento/)


<br>
<br>


# Função de Ativação

>### Todos os códigos abaixo foram feitos em Python com ajuda da lib NUMPY.

<br>

## Step (função degrau)
```
    def stepFunction(soma):
        if (soma >= 1):
            return 1
        return 0
```

## Sigmoid (função sigmoide)
*Medir uma probabilidade* (de 0 até 1)
```
    def sigmoidFuncion(soma):
        return 1 / (1 + np.exp(-soma))
```

## Hyperbolic tanget (função tangente hiperbólica)
*Medir uma probabilidade* (de -1 até 1)
```
    def tahnFunction(soma):
        return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))
```

## ReLU (rectified linear units)
*Mais usada em redes neurais profundas e convolucionais*
```
    def reluFunction(soma):
        if (soma >= 0):
            return soma
        return 0
```

## Linear
*Muito utilizada em redes de regressão, o que entra é igual o que sai*
```
    def linearFunction(soma):
        return soma
```

## Softmax
*Mais usada para retornar probabilidade com mais de uma classe*

*Usada na camada de saída*
```
    def softmaxFunction(x):
        ex = np.exp(x)
        return ex / ex.sum()
```

<br>
<br>


## Mais informações podem ser encontradas na [documentação do Keras](https://keras.io/api/layers/activations/)