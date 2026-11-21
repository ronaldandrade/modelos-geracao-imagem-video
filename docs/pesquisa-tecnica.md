# Pesquisa Técnica: A Engenharia por Trás da Geração de Imagem e Vídeo por IA

Este documento reúne a fundamentação matemática, decisões arquiteturais e comparativo das principais famílias de modelos generativos visuais do estado da arte (foco em 2023–2026).

---

## 1. A Evolução dos Paradigmas de Geração de Imagem

### 1.1 Do Pixel ao Latente: GANs vs VAEs vs Difusão

Historicamente, a síntese de imagem operou sob quatro grandes paradigmas:

| Paradigma | Mecanismo Central | Vantagens | Principais Gargalos | Modelos Clássicos |
|---|---|---|---|---|
| **GANs** (2014–2021) | Jogo Minimax entre Gerador $G$ e Discriminador $D$ | Amostragem em 1 passo, imagens nítidas | Colapso de modo (*mode collapse*), instabilidade extrema no treino, difícil cobertura de distribuição | DCGAN, StyleGAN2/3, BigGAN |
| **VAEs / VQ-GAN** (2013–2021) | Compressão em espaço latente contínuo ou discretizado com codebook | Treino estável por máxima verossimilhança (ELBO) | Imagens borradas (regressão à média em VAE simples); VQ-GAN requer quantização vetorial pesada | VAE, VQ-VAE-2, VQGAN |
| **Modelos Autorregressivos** (2020–hoje) | Previsão sequencial do próximo token visual (raster scan) via Transformers | Escala excelente com parâmetros e dados | Custo computacional quadrático no número de tokens; geração sequencial lenta | Parti, DALL-E 1, LlamaGen, Muse |
| **Modelos de Difusão (DDPM / LDM)** (2020–hoje) | Destruição progressiva por ruído gaussiano e reconstrução reversa aprendida | Cobertura fantástica da distribuição, estabilidade de treino | Dezenas de passos de amostragem iterativa; trajetórias estocásticas curvas | DDPM, DDIM, Stable Diffusion 1.5/XL |
| **Flow Matching / Rectified Flow** (2022–hoje) | Interpolação linear direta (trajetória reta) no espaço de probabilidades via ODEs | Amostragem em poucos passos (10–28), dinâmica determinística rápida | Requer arquiteturas com grande capacidade (DiT) para estimar o campo de velocidade | Flux.1, SD3 / SD3.5, Wan2.1 |

---

### 1.2 A Matemática da Difusão Clássica (DDPM)

O modelo DDPM (*Denoising Diffusion Probabilistic Models*, Ho et al., 2020) divide a geração em dois processos:

#### Processo Direto (*Forward Process*)
Uma cadeia de Markov que adiciona ruído gaussiano a uma imagem limpa $x_0$ em $T$ passos discretos:

$$
q(x_t | x_{t-1}) = \mathcal{N}\left(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t \mathbf{I}\right)
$$

Pela propriedade de soma de variáveis normais independentes, qualquer passo $t$ pode ser amostrado diretamente de $x_0$:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

onde $\alpha_t = 1 - \beta_t$ e $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

#### Processo Reverso (*Reverse Process*)
A rede neural $\epsilon_\theta(x_t, t)$ é treinada para prever exatamente o ruído $\epsilon$ adicionado no instante $t$:

$$
\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

Na inferência, partindo de ruído puro $x_T \sim \mathcal{N}(0, \mathbf{I})$, remove-se iterativamente o ruído estimado ao longo de dezenas de passos:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

#### Por que o LDM (*Latent Diffusion Models*) mudou tudo?
Operar difusão diretamente em pixels de alta resolução ($512 \times 512 \times 3 = 786.432$ valores) exige memória e processamento inviáveis em GPUs domésticas. O LDM (Rombach et al., 2022 / Stable Diffusion) resolveu isso comprimindo a imagem através de um VAE pré-treinado com fator de escala $f=8$:

$$
x \in \mathbb{R}^{3 \times 512 \times 512} \xrightarrow{\mathcal{E}} z \in \mathbb{R}^{4 \times 64 \times 64}
$$

A difusão roda no tensor latente $z$, que tem apenas 16.384 valores — uma redução de **48 vezes** no volume de dados processados pela rede a cada passo.

---

### 1.3 O Novo Paradigma: Flow Matching e Rectified Flow

Embora o LDM tenha democratizado a difusão, os amostradores estocásticos (como SDEs) geram trajetórias curvas e erráticas no espaço latente. Para contornar esse problema, surgiu o **Flow Matching** (Lipman et al., 2022) e o **Rectified Flow** (Liu et al., 2022):

#### A Trajetória Retilínea
Em vez de difusão browniana, define-se uma interpolação linear direta entre a distribuição de ruído $x_1 \sim \mathcal{N}(0, \mathbf{I})$ e os dados limpos $x_0 \sim p_{\text{data}}$:

$$
x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]
$$

A derivada temporal dessa trajetória é simplesmente a velocidade constante:

$$
\frac{d x_t}{dt} = x_1 - x_0
$$

A rede neural $v_\theta(x_t, t)$ é treinada para prever esse campo de velocidade vetorial direto:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

#### Por que isso é superior?
1. **Trajetórias retas (*straight paths*):** O caminho entre o ruído e a imagem é uma linha reta. Integradores de EDOs (como Euler simples ou Heun) conseguem percorrer a trajetória em apenas 10 a 28 passos sem acumular desvios que estragam o resultado.
2. **Estabilidade de treino:** Sem schedules de ruído complexos ($\beta_t$ cosine ou linear).
3. **Adoção nos modelos líderes:** É a base matemática do **Flux.1** (Black Forest Labs), do **Stable Diffusion 3/3.5** e do **Wan2.1**.

---

### 1.4 A Mudança Arquitetural: U-Net vs DiT vs MMDiT

Durante anos, a U-Net convolucional foi a espinha dorsal de qualquer modelo de difusão. A partir de 2023 (Peebles & Xie, *Diffusion Transformers*), a indústria migrou para Transformers:

```
[Latente 2D / 3D] ──> [Patchify (Linear Projection)] ──> [Tokens Visuais]
                                                               │
[Prompt de Texto] ──> [Text Encoders (T5 / CLIP / MLLM)] ──> [Tokens de Texto]
                                                               │
                           ┌───────────────────────────────────┘
                           ▼
             ┌───────────────────────────┐
             │   MMDiT / DiT Blocks      │
             │  • RoPE (Positional)      │
             │  • Self/Cross Attention   │
             │  • AdaLN-Zero Modulation  │
             └───────────────────────────┘
                           │
                           ▼
                    [Campo de Velocidade / Ruído Predito]
```

- **U-Net:** Limitada por induções convolucionais locais; difícil de escalar parâmetros além de 2 a 3 bilhões sem instabilidades.
- **DiT (*Diffusion Transformer*):** Transforma latentes em uma sequência de patches (análogo ao ViT), processados por blocos padrão de Transformer. Segue as leis de escala (*scaling laws*) empíricas com previsibilidade.
- **MMDiT (*Multi-Modal DiT* - SD3 e Flux.1):** Resolve o problema da interferência mútua. Em vez de forçar tokens de texto e imagem pelos mesmos pesos desde a primeira camada, o MMDiT mantém dois fluxos independentes de transformações lineares que trocam informação apenas nas operações de atenção, preservando os detalhes visuais e a obediência semântica ao texto.

---

## 2. A Geração de Vídeo: A Quarta Dimensão

### 2.1 O Desafio Fundamental do Vídeo

Um vídeo é um tensor $V \in \mathbb{R}^{T \times H \times W \times C}$. Adicionar a dimensão temporal $T$ traz três desafios críticos:

1. **Explosão de Contexto:** Um clipe de 5 segundos a 24 fps tem $T = 120$ frames. Em resolução $720\text{p}$ ($1280 \times 720$), são mais de 330 milhões de valores em ponto flutuante por amostra bruta.
2. **Atenção Quadrática:** Em Transformers, a complexidade da atenção padrão escala com $\mathcal{O}((T \cdot H \cdot W)^2)$.
3. **Consistência Causal e Física:** A IA não possui um motor de física nem uma representação 3D do mundo; ela é um aproximador estatístico de correlações espaço-temporais. Isso gera os problemas típicos: membros fantasmas, deformações durante rotações de câmera e dinâmica de fluidos fisicamente impossível.

---

### 2.2 O Componente Crucial: 3D Causal VAE

Para tornar a difusão de vídeo computacionalmente viável, todos os modelos modernos utilizam um **3D Causal Variational Autoencoder**:

```
Vídeo Bruto (T x H x W x 3)
         │
         ▼ [3D Causal VAE Encoder]
         │ Comprime tempo (ex: 4x) e espaço (ex: 8x)
         ▼
Latente de Vídeo ( (T/4) x (H/8) x (W/8) x C_latent )
```

- **Compressão Espaço-Temporal:** Reduz a dimensão temporal (geralmente por fator 4 ou 8) e a espacial (fator 8). Um vídeo de 120 quadros $720 \times 1280$ vira um tensor latente de apenas $30 \times 90 \times 160 \times 16$.
- **Por que Causal?** As convoluções e atenções temporais no encoder/decoder só podem olhar para quadros passados e presentes, nunca para quadros futuros. Isso impede o vazamento de informação temporal e viabiliza geração progressiva ou streaming sem distorções no histórico.

---

### 2.3 Patches Espaço-Temporais 3D (*Spacetime Patches*)

Popularizados pelo Sora e adotados no Wan2.1, HunyuanVideo e CogVideoX:

O tensor latente $(T' \times H' \times W')$ é fatiado em cubos tridimensionais de tamanho $p_t \times p_h \times p_w$ (tipicamente $2 \times 2 \times 2$). Cada cubo é projetado linearmente para um vetor de dimensão $D$, tornando-se um token único da sequência do Transformer.

Número de tokens resultantes:

$$
N_{\text{tokens}} = \left(\frac{T'}{p_t}\right) \times \left(\frac{H'}{p_h}\right) \times \left(\frac{W'}{p_w}\right)
$$

---

## 3. Raio-X dos Principais Modelos de Imagem e Vídeo

| Modelo | Criador | Tipo | Backbone | Objetivo / Amostrador | Encoders de Texto | Inovações Notáveis |
|---|---|---|---|---|---|---|
| **Flux.1** (Schnell / Dev / Pro) | Black Forest Labs | Imagem | MMDiT (12B parâmetros) | Rectified Flow Matching (4 a 28 passos) | T5-XXL + CLIP ViT-L | Fluxos multimodais duplos, qualidade fotográfica extrema e renderização tipográfica precisa |
| **Stable Diffusion 3.5** | Stability AI | Imagem | MMDiT (2B e 8B) | Flow Matching | CLIP ViT-L, OpenCLIP bigG, T5-XXL | Blocos duplos com atenção compartilhada e alta fidelidade ao prompt |
| **Midjourney (v6 / v7)** | Midjourney | Imagem | Proprietário (DiT híbrido) | Difusão proprietária | LLM proprietária / recaptioning interno | Coerência estética superior, composição cinematográfica apurada |
| **DALL-E 3** | OpenAI | Imagem | U-Net / DiT latente | Difusão guiada | GPT-4V (recaptioner) + T5/CLIP | Recaptioning denso: prompts do usuário são reescritos por LLM para guiar a síntese |
| **Wan2.1** | Alibaba Tongyi Lab | Vídeo | Spatiotemporal DiT (1.3B e 14B) | Flow Matching | T5 Encoder | 3D Causal VAE (Wan-VAE), versão 1.3B roda em GPUs de 24 GB (RTX 4090), suporte I2V/T2V nativo |
| **HunyuanVideo** | Tencent | Vídeo | Dual-stream to Single-stream DiT (13B) | Flow Matching | MLLM (multimodal) + CLIP | Separação estrita de texto e vídeo nas camadas iniciais e fusão posterior; pesos abertos |
| **CogVideoX** | THUDM / Zhipu AI | Vídeo | Expert Transformer DiT (2B e 5B) | 3D VAE + Flow/Diffusion | T5-XXL | Atenção conjunta com blocos especialistas para texto e vídeo; 3D VAE causal eficiente |
| **Sora** | OpenAI | Vídeo | Spacetime DiT em escala massiva | Difusão latente em patches 3D | GPT-recaptioner + MLLM | Pioneiro na formulação de patches 3D latentes em resolução flexível de aspecto e duração |
| **Kling 1.5 / 2.0** | Kuaishou | Vídeo | 3D DiT proprietário | Flow Matching proprietário | MLLM multilíngue | Duração de até 2 a 3 minutos, simulação superior de movimentos humanos e mecânicos |
| **Runway Gen-3 Alpha** | Runway | Vídeo | Spatiotemporal DiT | Difusão latente | Text Encoder + Motion Controllers | Alta fidelidade em controle cinematográfico de câmera e consistência de iluminação |

---

## 4. O Mecanismo de Condicionamento de Texto

A qualidade do resultado visual depende diretamente da capacidade do modelo de compreender o prompt:

1. **CLIP (ViT-L/14):** Excelente para alinhamento geral de estilo e conceitos simples, mas tem contexto limitado (77 tokens) e falha com descrições espaciais complexas ou contagem de objetos.
2. **T5-XXL / ByT5:** Modelos de linguagem pura com 4,7 bilhões de parâmetros. Conseguem entender gramática detalhada, relações sintáticas intrincadas e grafia exata de palavras para renderização de texto na imagem.
3. **MLLM (Multimodal Large Language Models):** Utilizados em modelos de vídeo modernos (como HunyuanVideo). O encoder textual é treinado conjuntamente para entender descrições de movimentos temporais compostos ("a pessoa corre, tropeça e a garrafa rola para a esquerda").
4. **Classifier-Free Guidance (CFG):**
   Durante a inferência, a predição é interpolada entre a saída condicionada ao texto $v_\theta(x_t, t, c)$ e a saída incondicionada $v_\theta(x_t, t, \emptyset)$:

   $$
   \tilde{v}_\theta = v_\theta(x_t, t, \emptyset) + s \cdot \left( v_\theta(x_t, t, c) - v_\theta(x_t, t, \emptyset) \right)
   $$

   Onde $s$ é a escala de guia (*guidance scale*). Aumentar $s$ melhora a obediência ao texto, mas valores excessivos introduzem saturação e artefatos de contraste.

---

## 5. Como os Modelos São Avaliados e Por Que Erram a Física

### 5.1 As Métricas de Avaliação

- **FID (*Fréchet Inception Distance*):** Mede a distância entre as distribuições estatísticas de ativação de uma rede Inception-v3 para imagens reais e imagens geradas. Quanto menor, mais realista.
- **CLIP Score:** Similaridade de cosseno entre o vetor de embedding do texto e o vetor da imagem gerada no espaço do CLIP.
- **FVD (*Fréchet Video Distance*):** Extensão do FID para sequências temporais usando ativações da rede I3D.
- **VBench:** Suíte moderna para vídeo que segmenta avaliação em 16 eixos: consistência de identidade de sujeito, consistência de fundo, amplitude de movimento, fidelidade temporal, qualidade estética e adesão física.

### 5.2 O Limite da Simulação Física
Por que modelos de vídeo ainda "alucinam" leis da física?
- Os modelos **não possuem simuladores newtonianos nem cálculo diferencial de fluidos**. Eles operam sobre densidades de probabilidade no espaço latente.
- Quando um objeto gira ou passa por trás de outro, o modelo prevê a configuração de patches mais provável segundo o dataset de treino. Se o dataset tem poucas variações daquele ângulo específico ou daquele movimento elástico, a minimização do erro pode gerar interpolações plásticas, membros duplicados ou desaparecimento de objetos.
