# 🧠 Transformer para Tradução EN → PT

### Laboratório 05 — Treinamento Fim-a-Fim

## 📌 Visão Geral

Este projeto implementa, do zero, uma arquitetura **Transformer** para a tarefa de **tradução automática de inglês (EN) para português (PT)**, utilizando PyTorch.

O objetivo principal é demonstrar o funcionamento completo de um modelo Transformer, incluindo:

* preparação de dados
* tokenização
* construção do modelo (encoder + decoder)
* treinamento
* inferência (geração de texto)

O modelo foi treinado utilizando um subconjunto do dataset **opus_books**, permitindo validação rápida e análise de comportamento (overfitting).

---

## ⚙️ Tecnologias Utilizadas

* **Python 3**
* **PyTorch** → construção e treinamento do modelo
* **Hugging Face Datasets** → carregamento do dataset
* **Hugging Face Transformers** → tokenização (BERT tokenizer)
* **NumPy / Math** → operações matemáticas auxiliares

---

## 🤖 Uso de Inteligência Artificial

Ferramentas de IA foram utilizadas como **apoio educacional e produtivo**, incluindo:

* compreensão do problema proposto no laboratório
* entendimento da arquitetura Transformer
* auxílio na interpretação da sintaxe das bibliotecas (PyTorch, Hugging Face)
* esclarecimento de dúvidas conceituais (atenção, masking, embeddings, etc.)
* organização e padronização do código
* geração de textos auxiliares, como este README

A IA foi utilizada como **ferramenta de aprendizado**, não substituindo o entendimento do funcionamento do modelo.

---

## 🏗️ Estrutura do Projeto

```
├── utils.py           # Máscaras e codificação posicional
├── add_norm.py        # Residual + LayerNorm
├── ffn.py             # Feed Forward Network
├── attention.py       # Multi-Head Attention
├── encoder.py         # Encoder do Transformer
├── decoder.py         # Decoder do Transformer
├── transformer.py     # Modelo completo
├── dataset.py         # Preparação e DataLoader
├── train.py           # Treinamento
└── inference.py       # Inferência e testes
```

---

## 🧠 Arquitetura do Modelo

O modelo segue a arquitetura clássica do Transformer:

### 🔹 Encoder

* Multi-Head Self-Attention
* Feed Forward Network
* Residual Connections + Layer Normalization

### 🔹 Decoder

* Masked Self-Attention
* Cross-Attention (com encoder)
* Feed Forward Network
* Residual Connections + Layer Normalization

### 🔹 Componentes principais

* Embeddings
* Positional Encoding
* Máscaras (causal e padding)
* Projeção final para vocabulário

---

## 🔄 Fluxo de Execução

### 1. 📥 Carregamento de Dados

* Dataset: `Helsinki-NLP/opus_books (en-pt)`
* Subconjunto reduzido (1000 exemplos)
* Tokenização com BERT

### 2. 🔤 Tokenização

* Conversão de texto → IDs
* Inclusão de tokens especiais (`[CLS]`, `[SEP]`)
* Truncamento para tamanho máximo

### 3. 📦 DataLoader

* Padding dinâmico
* Batch training

### 4. 🏋️ Treinamento

* Loss: CrossEntropyLoss
* Otimizador: Adam
* Teacher forcing no decoder
* Gradient clipping

### 5. 🔮 Inferência

* Decodificação autoregressiva
* Geração token a token
* Parada ao encontrar token de fim

---

## ▶️ Como Executar

### 🔹 Treinamento

```bash
python train.py
```

### 🔹 Inferência / Testes

```bash
python inference.py
```

---

## 📊 Resultados

Durante o treinamento:

* o modelo aprende a reproduzir o dataset
* ocorre redução progressiva da loss
* é possível observar overfitting devido ao pequeno tamanho do dataset

A inferência demonstra a capacidade do modelo de gerar traduções token a token.

---

## ⚠️ Limitações

* Dataset pequeno → alto risco de overfitting
* Modelo reduzido (menos camadas e dimensões)
* Não otimizado para produção

---

