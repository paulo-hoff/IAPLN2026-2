# Atividade Avaliativa — Análise de Sentimento com PLN

> Este repositório contém o enunciado e os dados. **Não faça commits aqui diretamente:** crie o seu próprio repositório conforme a seção 0.

**Disciplina:** Fundamentos de Inteligência Artificial
**Conteúdo:** Aula 11 (PLN), com apoio das Aulas 6–8 (ML supervisionado, métricas e validação cruzada)
**Modalidade:** individual · **Duração máxima:** 2 horas · **Entrega:** repositório no GitHub
**Linguagem:** Python 3.10+, execução local

---

## 1. Contexto

Uma loja online recebe centenas de avaliações de clientes por dia e quer saber, automaticamente, se cada texto é **positivo** ou **negativo**. Você foi contratado(a) para construir um protótipo de classificador de sentimento e, principalmente, para **dizer com honestidade o quanto ele pode ser confiável**.

Neste repositório você encontra:

- `data/avaliacoes.csv` — 260 avaliações em português (`id, texto, sentimento`), balanceadas (130 positivas e 130 negativas).
- `data/casos_dificeis.csv` — 13 frases escolhidas para testar os limites do modelo (`id, texto, sentimento_esperado, tipo`). **Estas frases NÃO entram no treino nem na escolha do modelo.**

> O dataset é didático e parte dele foi gerada por padrões. Os resultados serão mais otimistas do que seriam com dados reais. Isso é discutido no relatório.

## 2. Objetivos de aprendizagem

Ao final você deverá ser capaz de:

1. Pré-processar texto (normalização, tokenização, stopwords) justificando cada decisão.
2. Representar texto com Bag of Words e TF-IDF e comparar as duas representações.
3. Treinar um classificador supervisionado e avaliá-lo com métricas adequadas.
4. Evitar **vazamento de dados** (data leakage).
5. Usar validação cruzada para escolher o modelo e o conjunto de teste só para a avaliação final.
6. Analisar erros e identificar limites da abordagem (negação, ironia, frases mistas).

## 3. Regras e ambiente

- Bibliotecas permitidas: `pandas`, `numpy`, `scikit-learn`, `matplotlib` (opcional), `re` e a biblioteca padrão.
- **Não use** `transformers`, `nltk`, `spaCy` nem modelos pré-treinados nesta atividade.
- Use `random_state=42` em tudo que envolva aleatoriedade.
- O projeto deve rodar do zero com `python main.py` a partir de um ambiente limpo (`pip install -r requirements.txt`).

### Uso de IA generativa

É **proibido** usar IA para gerar o código ou o texto do relatório. A declaração de uso no final do relatório é obrigatória. O professor poderá pedir uma explicação oral do seu código e dos seus números. Resultados no relatório que não batem com os arquivos de `resultados/` serão tratados como inconsistência grave.

## 0. Como começar

1. Clone este repositório para ter o enunciado e os dados:
   ```bash
   git clone <URL-DESTE-REPOSITORIO>
   ```
2. Crie **um repositório novo seu** no GitHub (ex.: `sentimento-pln`), sem o histórico deste. Copie para ele a pasta `data/` e o arquivo `RELATORIO_modelo.md`.
3. Todo o seu trabalho (código, resultados e relatório) fica no **seu** repositório, que é o que será entregue e avaliado. Informe a URL na plataforma da disciplina.

## 4. Estrutura obrigatória do seu repositório

```
sentimento-pln/
├── data/
│   ├── avaliacoes.csv
│   └── casos_dificeis.csv
├── src/                 # seus módulos (ex.: preprocessamento, modelos, avaliacao)
├── resultados/          # gerado pelo seu código (não escreva à mão)
├── main.py              # executa tudo e gera os arquivos de resultados/
├── README.md            # como instalar e executar (curto)
├── RELATORIO.md         # mini relatório (seção 8)
├── requirements.txt
└── .gitignore
```

Faça **no mínimo 5 commits** com mensagens significativas (ex.: "adiciona pré-processamento", "compara BoW e TF-IDF"). Um único commit no final será penalizado.

## 5. Plano de tempo sugerido (120 min)

| Etapa | Tempo |
|---|---|
| 0. Setup do repositório e ambiente | 10 min |
| 1. Exploração dos dados | 10 min |
| 2. Pré-processamento | 15 min |
| 3. Vetorização e modelos (4 configurações) | 30 min |
| 4. Validação cruzada e escolha do modelo | 10 min |
| 5. Avaliação final, erros e palavras influentes | 15 min |
| 6. Casos difíceis | 10 min |
| 7. Relatório | 20 min |
| 8. Revisão e `git push` | — (margem) |

## 6. Etapas

### Etapa 0 — Setup

Crie o repositório, a estrutura de pastas, o ambiente virtual e o `requirements.txt` com as versões que você usou. Confirme que os dois CSVs estão em `data/`. Faça o primeiro commit.

### Etapa 1 — Exploração

Carregue `avaliacoes.csv` e produza, em `resultados/exploracao.txt`:

- número de linhas e distribuição das classes;
- tamanho médio dos textos (em palavras) por classe;
- 3 exemplos de cada classe.

*Pergunta para si mesmo(a):* o dataset está balanceado? Isso muda a métrica que você deve olhar?

### Etapa 2 — Pré-processamento

Escreva uma função que receba um texto e devolva o texto tratado. Ela deve, no mínimo:

- converter para minúsculas;
- remover pontuação e números, mantendo letras acentuadas;
- permitir **ligar/desligar** a remoção de stopwords.

**Você define a sua própria lista de stopwords** (mínimo de 15 palavras) e deve **justificar no relatório** o tratamento dado à palavra **"não"**. Pense no que acontece com "não gostei" se a palavra for removida.

Exemplo apenas da *assinatura* de uma ferramenta útil (não é uma solução):

```python
import re
re.findall(r"[a-záàâãéêíóôõúçü]+", texto)   # o que isso devolve? teste.
```

### Etapa 3 — Vetorização e modelos

Compare **quatro configurações**:

| Config | Stopwords | Vetorizador |
|---|---|---|
| A | mantidas | Bag of Words |
| B | mantidas | TF-IDF |
| C | removidas | Bag of Words |
| D | removidas | TF-IDF |

Requisitos:

- Mesmo classificador nas quatro (use Regressão Logística).
- **Uma única divisão treino/teste** para todas as configurações: 75% treino / 25% teste, estratificada, `random_state=42`.
- O vetorizador deve ser **ajustado (fit) apenas com dados de treino**. Veja as Aulas 6 e 8: ajustar o vetorizador com o dataset inteiro é vazamento. Pesquise como encadear vetorizador e modelo de forma que isso seja garantido durante a validação cruzada.
- Em cada configuração, registre o tamanho do vocabulário aprendido no treino.

Assinaturas úteis (apenas referência de nomes, você precisa conectá-las):

```python
CountVectorizer(...)         TfidfVectorizer(...)
LogisticRegression(max_iter=1000)
train_test_split(..., stratify=..., random_state=42)
```

### Etapa 4 — Validação cruzada e escolha do modelo

- Aplique validação cruzada estratificada de **5 folds** (`shuffle=True`, `random_state=42`) **somente nos dados de treino**, para cada configuração.
- Registre média e desvio padrão da acurácia e do F1.
- Escolha a configuração vencedora **pelo resultado da validação cruzada**, nunca pelo teste.
- Salve a tabela em `resultados/comparacao.csv` com as colunas: `config, stopwords, vetorizador, vocabulario, acc_cv_media, acc_cv_desvio, f1_cv_media, f1_cv_desvio`.

### Etapa 5 — Avaliação final, erros e palavras influentes

Com a configuração escolhida, treine no conjunto de treino inteiro e avalie **uma única vez** no teste:

- matriz de confusão (salve em `resultados/matriz_confusao.png` ou `.csv`);
- `classification_report` (salve em `resultados/relatorio_classificacao.txt`);
- `resultados/erros.csv` com os textos que o modelo errou (texto, real, previsto, probabilidade);
- `resultados/palavras_influentes.csv` com as **10 palavras que mais empurram para positivo** e as **10 que mais empurram para negativo**, com seus coeficientes.

### Etapa 6 — Casos difíceis

Aplique o modelo final a `casos_dificeis.csv` e salve `resultados/casos_dificeis_predicoes.csv` com: texto, tipo, esperado, previsto e probabilidade de positivo.

Atenção: há frases do tipo **misto** e **neutro** cujo esperado não é "positivo" nem "negativo". Seu modelo só conhece duas classes. Decida como apresentar isso e comente no relatório.

## 7. Entregas (checklist)

- [ ] Repositório público (ou privado com acesso para o professor) com ≥ 5 commits
- [ ] `python main.py` gera todos os arquivos de `resultados/`
- [ ] `requirements.txt` e `README.md` com instruções de execução
- [ ] Sem vazamento de dados (vetorizador ajustado só no treino)
- [ ] Modelo escolhido por validação cruzada
- [ ] `RELATORIO.md` completo, com todas as respostas entre 4 e 12 linhas
- [ ] Declaração de uso de IA preenchida

**Bônus (até +0,5, nota máxima 10):** comparar com n-gramas (1,2) **ou** implementar um baseline por léxico com tratamento de negação, e discutir no relatório se melhorou. O bônus só vale se o restante estiver correto.

## 8. Mini relatório

Copie o arquivo `RELATORIO_modelo.md` (deste repositório) para a raiz do **seu** repositório com o nome `RELATORIO.md` e responda as 6 perguntas. **Cada resposta deve ter de 4 a 12 linhas**, escritas com suas palavras e com números extraídos de `resultados/`. Respostas genéricas, sem números ou fora do intervalo de linhas perdem pontos.

## 9. Critérios de avaliação (10,0 pontos)

| Critério | Pontos |
|---|---|
| Estrutura do repositório, reprodutibilidade e commits | 1,0 |
| Qualidade do código (organização, funções, legibilidade) | 1,0 |
| Pré-processamento e justificativas | 1,0 |
| Vetorização e modelos sem vazamento | 1,5 |
| Avaliação (CV, teste final, métricas) | 1,5 |
| Análises geradas (erros, palavras, casos difíceis) | 1,0 |
| Relatório | 3,0 |

**Descontos:** vazamento de dados (−1,0 a −1,5); escolha do modelo pelo teste (−1,0); resultados não reproduzíveis (−0,5); relatório com números que não batem com `resultados/` (−1,0); ausência da declaração de uso de IA (−0,5).
