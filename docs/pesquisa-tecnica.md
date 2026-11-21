# Pesquisa Técnica: A Engenharia por Trás da Geração de Imagem e Vídeo por IA

Este documento reúne a fundamentação matemática, decisões arquiteturais e comparativo das principais famílias de modelos generativos visuais do estado da arte (foco em 2023–2026). Revisado em 20/11/2026.

---

## 1. A Evolução dos Paradigmas de Geração de Imagem

### 1.1 Do Pixel ao Latente: GANs vs VAEs vs Difusão

Historicamente, a síntese de imagem passou pelos seguintes paradigmas:

| Paradigma | Mecanismo Central | Vantagens | Principais Gargalos | Modelos Clássicos |
|---|---|---|---|---|
| **GANs** (2014–2021) | Jogo Minimax entre Gerador $G$ e Discriminador $D$ | Amostragem em 1 passo, imagens nítidas | Colapso de modo (*mode collapse*), instabilidade no treino, difícil cobertura de distribuição | DCGAN, StyleGAN2/3, BigGAN |
| **VAEs / VQ-GAN** (2013–2021) | Compressão em espaço latente contínuo ou discretizado com codebook | Treino estável por máxima verossimilhança (ELBO) | Imagens borradas (regressão à média em VAE simples); VQ-GAN requer quantização vetorial | VAE, VQ-VAE-2, VQGAN |
| **Transformers sobre tokens visuais** (2020–hoje) | Autorregressivos: próximo token visual em ordem raster. Mascarados: vários tokens em paralelo | Escala bem com parâmetros e dados | Custo quadrático no número de tokens; geração autorregressiva sequencial lenta | Autorregressivos: DALL-E 1, Parti, LlamaGen. Mascarados: MaskGIT, Muse |
| **Modelos de Difusão (DDPM / LDM)** (2020–hoje) | Destruição progressiva por ruído gaussiano e reconstrução reversa aprendida | Ótima cobertura da distribuição, estabilidade de treino | Dezenas de passos de amostragem (1000 no DDPM original; 20–50 com DDIM/DPM-Solver) | DDPM, DDIM, Stable Diffusion 1.5/XL |
| **Flow Matching / Rectified Flow** (2022–hoje) | Regressão de um campo de velocidade ao longo de caminhos condicionais retos entre ruído e dado; amostragem por ODE | Alvo de treino simples; boa qualidade com menos passos (tipicamente 28–50 sem destilação, 1–4 com destilação) | Campo marginal ainda curvo; requer backbones grandes (DiT) | SD3 / SD3.5, Flux.1, Wan2.1, HunyuanVideo |

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
A rede neural $\epsilon_\theta(x_t, t)$ é treinada para prever o ruído $\epsilon$ adicionado no instante $t$:

$$
\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

Na inferência, partindo de ruído puro $x_T \sim \mathcal{N}(0, \mathbf{I})$, remove-se iterativamente o ruído estimado:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

O DDPM original percorre os $T = 1000$ passos. O DDIM (Song et al., 2020) reformula a amostragem como uma ODE determinística e reduz para 20–50 passos sem retreinar a rede.

#### Por que o LDM (*Latent Diffusion Models*) mudou tudo?
Difusão em pixels funciona (o DDPM original já gerava 256×256 em LSUN e CelebA-HQ; o ADM chegou a 512×512), mas o custo de treino e inferência cresce rápido com a resolução, porque a rede inteira roda dezenas de vezes por amostra sobre $512 \times 512 \times 3 = 786.432$ valores. O LDM (Rombach et al., 2022 / Stable Diffusion) resolveu isso comprimindo a imagem através de um VAE pré-treinado com fator de escala $f=8$:

$$
x \in \mathbb{R}^{3 \times 512 \times 512} \xrightarrow{\mathcal{E}} z \in \mathbb{R}^{4 \times 64 \times 64}
$$

A difusão roda no tensor latente $z$, que tem apenas 16.384 valores: uma redução de **48 vezes** no volume de dados processados pela rede a cada passo.

---

### 1.3 O Novo Paradigma: Flow Matching e Rectified Flow

Em paralelo, surgiram o **Flow Matching** (Lipman et al., 2022) e o **Rectified Flow** (Liu et al., 2022):

#### A Trajetória Retilínea (condicional)
Para cada par (ruído $x_1 \sim \mathcal{N}(0, \mathbf{I})$, dado $x_0 \sim p_{\text{data}}$), define-se uma interpolação linear:

$$
x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]
$$

A derivada temporal dessa trajetória é a velocidade constante:

$$
\frac{d x_t}{dt} = x_1 - x_0
$$

A rede neural $v_\theta(x_t, t)$ é treinada para regredir essa velocidade:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

#### O que isso garante e o que não garante
1. **Só os caminhos condicionais são retos.** O ótimo da perda é $v^*(x, t) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$, a média das velocidades de todos os pares que passam por $x$. Esse campo marginal é curvo quando $p_{\text{data}}$ tem vários modos, e a perda tem um piso irredutível maior que zero.
2. **Reflow endireita as trajetórias.** O Rectified Flow original propõe retreinar sobre pares (ruído, amostra) gerados pelo próprio modelo; as trajetórias ficam quase retas e a amostragem em 1–2 passos se torna viável.
3. **Na prática:** o alvo de velocidade é simples e dispensa schedules de ruído elaborados, e a qualidade com poucos passos tende a ser melhor que a da difusão com DDIM. O SD3 (Esser et al., 2024) mostrou que o Rectified Flow com amostragem de $t$ concentrada nos tempos intermediários (*logit-normal*) supera as formulações de difusão em escala. Amostragem em 1–4 passos em produção (Flux.1 Schnell, SD3.5 Turbo) vem de destilação.
4. **Adoção:** SD3/SD3.5, Flux.1, Wan2.1 e HunyuanVideo.

O experimento em `src/toy_diffusion_vs_flow.py` (8 gaussianas em 2D) mostra isso. A retidão é o comprimento do caminho dividido pela corda (reta = 1,0), e a precisão é a fração de amostras a menos de 0,3 de um centro:

| Modelo | Retidão | Precisão, 1 passo | 5 passos | 10 passos | 50 passos |
|---|---|---|---|---|---|
| DDPM + DDIM | 1,63 | 0,01 | 0,47 | 0,73 | 0,85 |
| Rectified Flow | 1,74 | 0,00 | 0,74 | 0,89 | 0,95 |
| Rectified Flow + 1 reflow | 1,00 | 0,78 | 0,88 | 0,91 | 0,93 |

---

### 1.4 A Mudança Arquitetural: U-Net vs DiT vs MMDiT

Durante anos, a U-Net convolucional foi a espinha dorsal dos modelos de difusão. Com o DiT (Peebles & Xie, arXiv dez/2022, ICCV 2023), a indústria migrou para Transformers:

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
             │  • Self/Joint Attention   │
             │  • AdaLN-Zero Modulation  │
             └───────────────────────────┘
                           │
                           ▼
           [Velocidade / Ruído / v predito]
```

- **U-Net:** viés indutivo convolucional local. As maiores ficaram na faixa de 2–3B parâmetros (a do SDXL tem 2,6B), sem uma lei de escala clara.
- **DiT (*Diffusion Transformer*):** transforma latentes em uma sequência de patches (análogo ao ViT), processados por blocos de Transformer com condicionamento por AdaLN-Zero. A qualidade (FID) melhora de forma previsível com mais GFLOPs.
- **MMDiT (*Multi-Modal DiT*, SD3):** texto e imagem têm pesos separados (projeções e MLPs) em **todos** os blocos e trocam informação apenas na atenção conjunta, em que as duas sequências são concatenadas.
- **Flux.1:** híbrido, com 19 blocos de fluxo duplo (estilo MMDiT) seguidos de 38 blocos de fluxo único, com pesos compartilhados. O HunyuanVideo usa o mesmo padrão dual → single stream.

---

## 2. A Geração de Vídeo: A Quarta Dimensão

### 2.1 O Desafio Fundamental do Vídeo

Um vídeo é um tensor $V \in \mathbb{R}^{T \times H \times W \times C}$. Adicionar a dimensão temporal $T$ traz três desafios críticos:

1. **Explosão de contexto:** um clipe de cerca de 5 segundos a 24 fps tem 121 frames ($4n+1$, ver 2.2). Em 720p ($1280 \times 720$), são cerca de 335 milhões de valores por amostra bruta.
2. **Atenção quadrática:** em Transformers, a atenção completa escala com $\mathcal{O}(N^2 d)$, com $N \propto T \cdot H \cdot W$.
3. **Consistência causal e física:** o modelo não possui um motor de física nem uma representação 3D explícita do mundo; ele é um aproximador estatístico de correlações espaço-temporais. Isso gera os problemas típicos: membros fantasmas, deformações durante rotações de câmera e dinâmica de fluidos fisicamente impossível.

---

### 2.2 O Componente Crucial: 3D Causal VAE

A maioria dos modelos de vídeo abertos recentes (Wan2.1, HunyuanVideo, CogVideoX) usa um **3D Causal Variational Autoencoder**:

```
Vídeo Bruto (T x H x W x 3), T = 4n + 1
         │
         ▼ [3D Causal VAE Encoder]
         │ 1º quadro sozinho; demais em grupos de 4 (tempo) e 8x (espaço)
         ▼
Latente de Vídeo ( (1 + (T-1)/4) x (H/8) x (W/8) x 16 )
```

- **Compressão espaço-temporal:** fator 4 no tempo e 8 no espaço (Wan-VAE e o VAE do HunyuanVideo, ambos com 16 canais). Um vídeo de 121 quadros $720 \times 1280$ vira um latente de $31 \times 90 \times 160 \times 16$.
- **Por que causal?** As convoluções temporais só olham para quadros passados e presentes. Com isso, o primeiro quadro é codificado como uma imagem comum, e imagens e vídeos compartilham o mesmo espaço latente. Isso permite treino conjunto imagem + vídeo, image-to-video e codificação de vídeos longos em blocos com cache de features.

---

### 2.3 Patches Espaço-Temporais (*Spacetime Patches*)

Popularizados pelo relatório do Sora (que não publicou detalhes da arquitetura) e usados em Wan2.1, HunyuanVideo e CogVideoX:

O tensor latente $(T' \times H' \times W')$ é fatiado em blocos de tamanho $p_t \times p_h \times p_w$, e cada bloco é projetado linearmente para um vetor de dimensão $D$:

$$
N_{\text{tokens}} = \left\lceil\frac{T'}{p_t}\right\rceil \times \frac{H'}{p_h} \times \frac{W'}{p_w}
$$

- **Wan2.1 e HunyuanVideo:** $p = (1, 2, 2)$, sem agrupamento temporal no patch. Para o latente acima, são $31 \times 45 \times 80 = 111.600$ tokens.
- **Patch temporal 2 (ex.: CogVideoX 1.5):** $p = (2, 2, 2)$ resulta em $16 \times 45 \times 80 = 57.600$ tokens.
- **Comparação com pixels em patches $1 \times 16 \times 16$** (435.600 tokens): a sequência fica 3,9× menor com (1,2,2) e 7,6× menor com (2,2,2); a matriz de atenção fica 15× e 57× menor, respectivamente. Ver `src/spacetime_patching.py`.

---

## 3. Raio-X dos Principais Modelos de Imagem e Vídeo

"N/D" = não divulgado pelo fabricante. Versões atualizadas até novembro de 2026.

| Modelo | Criador | Tipo | Backbone | Objetivo / Amostrador | Encoders de Texto | Notas |
|---|---|---|---|---|---|---|
| **Flux.1** (Schnell / Dev / Pro) | Black Forest Labs | Imagem | DiT híbrido: 19 blocos duplos + 38 simples (12B) | Rectified Flow; Dev ~28–50 passos; Schnell 1–4 passos (destilação adversarial) | T5-XXL + CLIP ViT-L | Renderização tipográfica precisa; Dev é destilado de guidance |
| **Stable Diffusion 3.5** | Stability AI | Imagem | MMDiT (Large 8B) / MMDiT-X (Medium 2.5B) | Rectified Flow | CLIP ViT-L, OpenCLIP bigG, T5-XXL | Atenção conjunta em todos os blocos; QK-normalization |
| **Midjourney (v6 / v7)** | Midjourney | Imagem | N/D | N/D | N/D | Forte em estética e composição |
| **Nano Banana / Pro / 2** | Google DeepMind | Imagem (geração e edição) | N/D (capacidade nativa de modelos Gemini) | N/D | Próprio Gemini | Gemini 2.5 Flash Image (ago/2025), Pro (nov/2025), Nano Banana 2 = Gemini 3.1 Flash Image (fev/2026); edição conversacional; marca d'água SynthID |
| **DALL-E 3** | OpenAI | Imagem | Difusão latente (U-Net) | Difusão | T5-XXL | Recaptioning com um captioner próprio treinado pela OpenAI; no ChatGPT, o GPT-4 reescreve o prompt. Substituído pela geração nativa do GPT-4o / gpt-image |
| **Wan2.1** | Alibaba Tongyi Lab | Vídeo | DiT espaço-temporal (1.3B e 14B), patch (1,2,2) | Flow Matching | umT5 (multilíngue) | Wan-VAE causal 4×8×8, 16 canais; 1.3B roda com ~8,2 GB de VRAM; T2V/I2V. Sucessores: Wan 2.2 (MoE) e 2.5 (com áudio) |
| **HunyuanVideo** | Tencent | Vídeo | DiT dual-stream → single-stream (13B), patch (1,2,2) | Flow Matching | MLLM + CLIP | VAE causal 4×8×8; pesos abertos. Sucessor: HunyuanVideo 1.5 |
| **CogVideoX** | Zhipu AI / THUDM | Vídeo | Expert Transformer (2B e 5B), expert adaptive LayerNorm | Difusão com v-prediction e zero-SNR | T5-XXL | Atenção 3D completa; VAE 3D causal |
| **Sora / Sora 2** | OpenAI | Vídeo (+ áudio no Sora 2) | DiT sobre patches espaço-temporais (detalhes N/D) | Difusão latente (detalhes N/D) | N/D (recaptioning estilo DALL-E 3) | Popularizou os patches espaço-temporais; Sora 2 (set/2025) gera áudio sincronizado |
| **Kling** | Kuaishou | Vídeo (+ áudio a partir do 2.6) | N/D (DiT 3D, segundo a empresa) | N/D | N/D | Clipes de 5–10 s, extensíveis até ~3 min; forte em movimento humano |
| **Runway Gen-4 / 4.5** | Runway | Vídeo | N/D | N/D | N/D | Consistência de personagem e cena por referência; controle de câmera |
| **Seedance 1.0 → 2.5** | ByteDance Seed | Vídeo (+ áudio a partir do 1.5 pro) | 1.5 pro: DiT de duas ramificações + módulo conjunto; 2.0: arquitetura unificada áudio-vídeo (detalhes N/D) | N/D | N/D | 1.0 (jun/2025) só vídeo; 1.5 pro (dez/2025) áudio-vídeo; 2.0 (fev/2026) referências texto/imagem/áudio/vídeo, 4–15 s; 2.5 (jul/2026) 30 s, até 50 referências |
| **Google Veo (2 / 3 / 3.1)** | Google DeepMind | Vídeo (+ áudio a partir do Veo 3) | N/D (difusão latente) | N/D | N/D | Veo 3 (mai/2025) com áudio nativo; forte em linguagem de câmera |

---

## 4. Fundação vs. Produto e o Condicionamento de Texto

### 4.1 Modelos fundacionais vs. camada de produto

Vale separar quem treina o backbone de quem constrói a esteira em volta dele.

1. **Modelos fundacionais (ex.: Seedance, Nano Banana, Veo):**
   - **Seedance 1.5 pro:** o relatório técnico (arXiv 2512.13507) descreve um DiT de duas ramificações, uma para os latentes de vídeo e outra para os de áudio, ligadas por um módulo cross-modal conjunto, o que dá sincronia de lábios e de efeitos sonoros.
   - **Seedance 2.0** (arXiv 2604.14148): descrito apenas como "arquitetura unificada" de geração conjunta. Ele e o 2.5 aceitam vídeos de referência para transferir movimento de câmera ou coreografia.
   - **Nano Banana:** é o modelo de imagem do Gemini, ou seja, geração e edição como capacidade nativa de um modelo multimodal. Não é um orquestrador.
   - **Áudio nativo** não é exclusividade do Seedance: Veo 3, Sora 2, Wan 2.5 e Kling 2.6 também geram áudio.

2. **Camada de produto e orquestração:**
   - Sites de terceiros (muitos usando "banana" no nome) e ferramentas como o ComfyUI encadeiam APIs desses modelos.
   - O diferencial técnico está no **pipeline de consistência de personagem e controle de cena**: encoders de identidade (IP-Adapter), inpainting guiado por máscara, controle de primeiro e último quadro (*first & last frame*) e filtros estéticos antes e depois da amostragem.

### 4.2 Encoders de texto e guidance

A qualidade do resultado visual depende diretamente da capacidade do modelo de compreender o prompt:

1. **CLIP (ViT-L/14):** bom para alinhamento geral de estilo e conceitos simples, mas tem contexto limitado (77 tokens) e falha com descrições espaciais complexas ou contagem de objetos.
2. **T5-XXL:** o encoder tem cerca de 4,7B parâmetros (o T5-XXL completo tem 11B). Entende gramática detalhada e relações sintáticas e ajuda na grafia exata de palavras renderizadas na imagem. Variantes multilíngues: umT5 (Wan) e ByT5 (nível de byte, usado para texto em imagem).
3. **MLLM (Multimodal Large Language Models):** usados como encoder de texto em modelos como o HunyuanVideo. Trazem melhor compreensão de instruções e descrições compostas de movimento ("a pessoa corre, tropeça e a garrafa rola para a esquerda").
4. **Classifier-Free Guidance (CFG):**
   Durante a inferência, a predição é interpolada entre a saída condicionada ao texto $v_\theta(x_t, t, c)$ e a saída incondicionada $v_\theta(x_t, t, \emptyset)$:

   $$
   \tilde{v}_\theta = v_\theta(x_t, t, \emptyset) + s \cdot \left( v_\theta(x_t, t, c) - v_\theta(x_t, t, \emptyset) \right)
   $$

   Onde $s$ é a escala de guia (*guidance scale*). Aumentar $s$ melhora a obediência ao texto, mas valores excessivos introduzem saturação e artefatos de contraste.

---

## 5. Como os Modelos São Avaliados e Por Que Erram a Física

### 5.1 As Métricas de Avaliação

- **FID (*Fréchet Inception Distance*):** distância de Fréchet entre gaussianas ajustadas às ativações da Inception-v3 para imagens reais e geradas. Quanto menor, mais realista.
- **CLIP Score:** similaridade de cosseno entre o embedding do texto e o da imagem gerada no espaço do CLIP.
- **FVD (*Fréchet Video Distance*):** extensão do FID para vídeo usando ativações da rede I3D.
- **VBench:** suíte para vídeo com 16 dimensões, entre elas consistência de sujeito e de fundo, cintilação temporal, suavidade e amplitude de movimento, qualidade estética e de imagem, e aderência semântica (objetos, ações, cor, relações espaciais, estilo). Física e senso comum entram no **VBench-2.0**.

### 5.2 O Limite da Simulação Física
Por que modelos de vídeo ainda "alucinam" leis da física?
- Os modelos **não possuem simuladores newtonianos nem resolvem equações de fluidos**. Eles modelam densidades de probabilidade no espaço latente.
- Quando um objeto gira ou passa por trás de outro, o modelo prevê a configuração de patches mais provável segundo os dados de treino. Se o dataset tem poucas variações daquele ângulo ou movimento, a minimização do erro pode gerar interpolações plásticas, membros duplicados ou desaparecimento de objetos.

---

## Referências principais

- Ho et al., *Denoising Diffusion Probabilistic Models* (2020). arXiv:2006.11239
- Song et al., *Denoising Diffusion Implicit Models* (2020). arXiv:2010.02502
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022). arXiv:2112.10752
- Lipman et al., *Flow Matching for Generative Modeling* (2022). arXiv:2210.02747
- Liu et al., *Flow Straight and Fast: Rectified Flow* (2022). arXiv:2209.03003
- Peebles & Xie, *Scalable Diffusion Models with Transformers* (2022). arXiv:2212.09748
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3, 2024). arXiv:2403.03206
- Yang et al., *CogVideoX* (2024). arXiv:2408.06072
- Kong et al., *HunyuanVideo* (2024). arXiv:2412.03603
- Wan Team, *Wan: Open and Advanced Large-Scale Video Generative Models* (2025). arXiv:2503.20314
- ByteDance Seed, *Seedance 1.5 pro* (2025). arXiv:2512.13507
- ByteDance Seed, *Seedance 2.0* (2026). arXiv:2604.14148
