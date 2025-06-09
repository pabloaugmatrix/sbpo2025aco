import cupy as cp
import numpy as np

from src import dados
from src.funcao_objetivo.funcao_objetivo import funcao_objetivo


def construir_matriz_heuristica_gpu_vetorizada_pedidos(dados):
    # Pega os vetores dos pedidos corretamente
    pedidos = list(dados['armazem']['pedidos'].values())
    # Conversão explícita para numpy float64 (CPU)
    matriz_np = np.array(pedidos, dtype=np.float64)
    # Envia para GPU
    matriz_pedidos = cp.asarray(matriz_np)
    # Produto interno entre todos os vetores (N x N)
    produto_interno = matriz_pedidos @ matriz_pedidos.T
    # Normas dos vetores
    normas = cp.linalg.norm(matriz_pedidos, axis=1)
    # Produto externo das normas
    norma_matriz = cp.outer(normas, normas)
    # Similaridade do cosseno com tratamento de divisão por zero
    heuristica = cp.where(norma_matriz == 0, 0, 1/(1-(produto_interno / norma_matriz)))
    return heuristica

def construir_matriz_heuristica_gpu_vetorizada_corredores(dados):
    # Pega os vetores dos pedidos corretamente
    corredores = list(dados['armazem']['corredores'].values())
    # Conversão explícita para numpy float32 (CPU)
    matriz_np = np.array(corredores, dtype=np.float64)
    # Envia para GPU
    matriz_corredores = cp.asarray(matriz_np)
    # Produto interno entre todos os vetores (N x N)
    produto_interno = matriz_corredores @ matriz_corredores.T
    # Normas dos vetores
    normas = cp.linalg.norm(matriz_corredores, axis=1)
    # Produto externo das normas
    norma_matriz = cp.outer(normas, normas)
    # Similaridade do cosseno com tratamento de divisão por zero
    heuristica = cp.where(norma_matriz == 0, 0, 1/(1-(produto_interno / norma_matriz)))
    return heuristica

def construir_heuristica_gpu_vetorizada_corredores2(dados):
    num_itens = len(next(iter(dados['armazem']['pedidos'].values())))
    num_corredores = len(dados['armazem']['corredores'])

    # Vetor de demanda
    pedidos_wave = [dados['armazem']['pedidos'][p] for p in dados['wave']['pedidos']]
    matriz_pedidos = cp.asarray(pedidos_wave, dtype=cp.float32)
    demanda = cp.sum(matriz_pedidos, axis=0)

    # Matriz corredores x itens
    matriz_corredores = cp.zeros((num_corredores, num_itens), dtype=cp.float64)
    for idx, itens in enumerate(dados['armazem']['corredores'].values()):
        matriz_corredores[idx, itens] = 1.0

    # Matriz de cobertura combinada
    cobertura_combinada = (matriz_corredores @ demanda)  # (N_corredores,)

    # Matriz heurística: soma das coberturas de cada par
    heuristica = cobertura_combinada[:, None] + cobertura_combinada[None, :]

    return heuristica  # Matriz (N_corredores x N_corredores)


def construir_matriz_heuristica_manhattan_gpu_vetorizada3(dados):
    # Pega os vetores dos pedidos corretamente
    corredores = list(dados['armazem']['corredores'].values())

    # Conversão para numpy (CPU) e depois para cupy (GPU)
    matriz_corredores = cp.asarray(np.array(corredores, dtype=np.float64))

    # Calcula a distância de Manhattan entre todos os pares
    n = matriz_corredores.shape[0]
    heuristica = cp.zeros((n, n), dtype=cp.float64)

    # Versão vetorizada para calcular a matriz de distâncias
    for i in range(n):
        heuristica[i, :] = cp.sum(cp.abs(matriz_corredores - matriz_corredores[i, :]), axis=1)

    # Transforma distância em similaridade (quanto menor a distância, maior a similaridade)
    # Adicionamos 1 para evitar divisão por zero e inverter a relação
    heuristica = 1 / (1 + heuristica)

    # Preenche a diagonal com 1 (distância zero para o mesmo elemento)
    cp.fill_diagonal(heuristica, 1)

    return heuristica


def construir_solucao_pedidos(dados, feromonio_mat, heuristica_mat, alfa, beta):
    atual = np.random.choice(list(dados['armazem']['pedidos'].keys()))
    caminho = [int(atual)]
    vertices_nao_visitados = [i for i in range(len(dados['armazem']['pedidos'])) if i != atual]
    dados['wave']['pedidos'].append(atual)
    dados['wave']['tamanho'] += sum(dados['armazem']['pedidos'][atual])
    while vertices_nao_visitados:
        pontuacoes = []
        for vertice in vertices_nao_visitados:
            feromonio = feromonio_mat[atual, vertice]
            heuristica = heuristica_mat[atual, vertice]
            pontuacao = (feromonio ** alfa) * (heuristica ** beta)
            pontuacoes.append(pontuacao)
        pontuacoes = cp.array(pontuacoes)
        probabilidades = pontuacoes / pontuacoes.sum()
        # Suponha que probabilidades seja cupy.ndarray e soma 1
        prob_cumsum = cp.cumsum(probabilidades)
        r = cp.random.random()

        # Busca o índice onde a soma acumulada excede r
        indice = cp.searchsorted(prob_cumsum, r)

        proximo = vertices_nao_visitados[int(indice.get())]  # Pega da lista de labels (CPU)

        if (sum(dados['armazem']['pedidos'][proximo]) + dados['wave']['tamanho']) > dados['wave']['limite_superior']:
            return caminho
        else:
            caminho.append(proximo)
            dados['wave']['pedidos'].append(proximo)
            dados['wave']['tamanho'] += sum(dados['armazem']['pedidos'][proximo])
            atual = proximo
            vertices_nao_visitados.remove(proximo)
    return caminho

def demanda_atendida(demanda, oferta):
    return cp.all(demanda <= oferta)  # Para arrays CuPy/NumPy


def construir_solucao_corredores(dados, feromonio_mat, heuristica_mat, alfa, beta):
    # Pré-calcula demandas (convertendo tudo para CuPy)
    pedidos_wave = dados['wave']['pedidos']
    pedidos_armazem = dados['armazem']['pedidos']

    # Converte tudo para CuPy arrays
    lista_pedidos = [cp.array(pedidos_armazem[pedido], dtype=cp.float64) for pedido in pedidos_wave]
    demanda = cp.sum(cp.stack(lista_pedidos), axis=0)

    oferta = cp.zeros_like(demanda)
    corredores_keys = list(dados['armazem']['corredores'].keys())
    corredores_values = cp.array(list(dados['armazem']['corredores'].values()), dtype=cp.float64)

    # Escolha inicial
    atual = np.random.choice(corredores_keys)
    caminho = [int(atual)]
    vertices_nao_visitados = {k for k in corredores_keys if k != atual}
    dados['wave']['corredores'].append(atual)

    while vertices_nao_visitados:
        # Atualiza oferta
        oferta += corredores_values[atual]

        if cp.all(demanda <= oferta).get():
            return caminho

        # Cálculo das probabilidades
        vertices_disponiveis = list(vertices_nao_visitados)

        # Garante que os índices sejam inteiros
        indices = [int(v) for v in vertices_disponiveis]

        # Extrai valores das matrizes (convertendo para CuPy se necessário)
        feromonios = cp.asarray(feromonio_mat[atual, indices])
        heuristicas = cp.asarray(heuristica_mat[atual, indices])

        # Calcula pontuações
        pontuacoes = (feromonios ** alfa) * (heuristicas ** beta)

        # Normaliza e converte para CPU para o random.choice
        prob = cp.asnumpy(pontuacoes / cp.sum(pontuacoes))
        proximo = np.random.choice(vertices_disponiveis, p=prob)

        caminho.append(proximo)
        dados['wave']['corredores'].append(proximo)
        atual = proximo
        vertices_nao_visitados.remove(proximo)

    return caminho


def atualiza_feromonios(feromonio_mat, solucoes, valores, evaporacao, feromonio, ferom_min=0.1):
    # Aplica evaporação
    feromonio_mat *= (1 - evaporacao)

    # Garante um valor mínimo de feromônio (opcional)
    feromonio_mat[feromonio_mat < ferom_min] = ferom_min

    # Atualiza feromônios para cada solução
    for solucao, valor in zip(solucoes, valores):
        if valor <= 0:  # Evita divisão por zero ou valores negativos
            continue

        delta = feromonio / valor
        for i in range(len(solucao) - 1):
            # Verifica índices válidos
            if (0 <= solucao[i] < feromonio_mat.shape[0] and
                    0 <= solucao[i + 1] < feromonio_mat.shape[1]):
                feromonio_mat[solucao[i], solucao[i + 1]] += delta
                feromonio_mat[solucao[i + 1], solucao[i]] += delta

def ant_colony_support(dados, iteracoes, formigas, evaporacao, feromonio, alfa, beta):
    heuristica_mat = construir_matriz_heuristica_manhattan_gpu_vetorizada3(dados)
    feromonio_mat = np.ones(heuristica_mat.shape)
    melhor_solucao = None
    melhor_valor = 0
    iteracoes_sem_melhorar = 0
    for _ in range(iteracoes):
        ultimo_melhor_valor = melhor_valor
        if iteracoes_sem_melhorar > iteracoes * 0.02:
            break
        solucoes = []
        valores = []
        for _ in range(formigas):
            #print("formiga corredores")
            dados['wave']['corredores'] = []
            solucao = construir_solucao_corredores(dados, feromonio_mat, heuristica_mat, alfa, beta)
            valor = funcao_objetivo(dados)
            solucoes.append(solucao)
            valores.append(valor)
            #print(valor)
            if valor > melhor_valor:
                melhor_valor = valor
                melhor_solucao = solucao
        if melhor_valor <= ultimo_melhor_valor:
            iteracoes_sem_melhorar += 1
        else:
            iteracoes_sem_melhorar = 0
        atualiza_feromonios(feromonio_mat, solucoes, valores, evaporacao, feromonio)
    return melhor_solucao, melhor_valor


def ant_colony(dados, iteracoes, formigas, evaporacao, feromonio, alfa, beta):
    heuristica_mat_pedidos = construir_matriz_heuristica_gpu_vetorizada_pedidos(dados)
    heuristica_mat_corredores = construir_matriz_heuristica_gpu_vetorizada_corredores(dados)
    feromonio_mat_pedidos = np.ones(heuristica_mat_pedidos.shape)
    #feromonio_mat_corredores = np.ones(heuristica_mat_corredores.shape)
    #print("heuristica construida!")
    #print(heuristica_mat_pedidos)
    #stop = input("!")

    melhor_solucao = None
    corredores_para_melhor_solucao = None
    melhor_valor = 0
    iteracoes_sem_melhorar = 0
    for _ in range(iteracoes):
        ultimo_melhor_valor = melhor_valor
        if iteracoes_sem_melhorar > 10:
            break
        solucoes = []
        valores = []
        for _ in range(formigas):
            #print("formiga pedidos")
            dados['wave']['pedidos'] = []
            dados['wave']['tamanho'] = 0
            solucao = construir_solucao_pedidos(dados, feromonio_mat_pedidos, heuristica_mat_pedidos, alfa, beta)
            corredores_para_solucao, valor = ant_colony_support(dados, iteracoes, formigas, evaporacao, feromonio, alfa, beta)
            solucoes.append(solucao)
            valores.append(valor)
            #print(solucao)
            #print(corredores_para_solucao)
            #print(valor)
            if valor > melhor_valor:
                melhor_valor = valor
                melhor_solucao = solucao
                corredores_para_melhor_solucao = corredores_para_solucao
        if melhor_valor <= ultimo_melhor_valor:
            iteracoes_sem_melhorar += 1
        else:
            iteracoes_sem_melhorar = 0
        atualiza_feromonios(feromonio_mat_pedidos, solucoes, valores, evaporacao, feromonio)
        print(f'MELHOR VALOR: {melhor_valor}')

    print(melhor_valor)
    dados['wave']['pedidos'] = sorted(melhor_solucao)
    print(dados['wave']['pedidos'])
    dados['wave']['corredores'] = sorted(corredores_para_melhor_solucao)
    print(dados['wave']['corredores'])



