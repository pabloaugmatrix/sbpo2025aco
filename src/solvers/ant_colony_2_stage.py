from multiprocessing import Pool
import numpy as np
from src.funcao_objetivo.funcao_objetivo import funcao_objetivo
from multiprocessing import Pool, cpu_count


def converter_para_array(pedido):
    """Converte um pedido para array numpy, independente do formato"""
    if isinstance(pedido, (list, tuple, np.ndarray)):
        return np.array(pedido)
    elif isinstance(pedido, dict):
        if 'vetor' in pedido:
            return np.array(pedido['vetor'])
        elif 'vector' in pedido:
            return np.array(pedido['vector'])
        else:
            return np.array(list(pedido.values()))
    elif isinstance(pedido, (int, float)):
        return np.array([pedido])
    else:
        raise ValueError(f"Tipo de pedido não suportado: {type(pedido)}")


def calcular_linha_matriz(args):
    i, pedidos, normas = args
    A = pedidos[i]
    norm_A = normas[i]

    linha = np.zeros(len(pedidos))
    for j in range(len(pedidos)):
        B = pedidos[j]
        norm_B = normas[j]

        if norm_A == 0 or norm_B == 0:
            cosine = 0
        else:
            cosine = np.dot(A, B) / (norm_A * norm_B)
        linha[j] = 1 / (1 - cosine + 1e-10)
    return i, linha


def construir_matriz_heuristica_pedidos_paralelo(dados):
    if not isinstance(dados, dict) or 'armazem' not in dados or 'pedidos' not in dados['armazem']:
        raise ValueError("Estrutura de dados inválida. Deve conter 'armazem']['pedidos'")

    try:
        list_pedidos = [converter_para_array(pedido) for pedido in dados['armazem']['pedidos']]
        pedidos = np.vstack(list_pedidos)
    except Exception as e:
        raise ValueError(f"Falha ao converter pedidos: {str(e)}")

    if len(pedidos.shape) != 2:
        raise ValueError("Os pedidos devem formar uma matriz 2D")

    quantidade_pedidos = pedidos.shape[0]
    normas = np.linalg.norm(pedidos, axis=1)

    heuristica_mat = np.zeros((quantidade_pedidos, quantidade_pedidos))

    num_processos = max(1, cpu_count() // 2)

    with Pool(processes=num_processos) as pool:
        args = [(i, pedidos, normas) for i in range(quantidade_pedidos)]
        resultados = pool.map(calcular_linha_matriz, args)

    for i, linha in resultados:
        heuristica_mat[i, :] = linha

    heuristica_mat = np.nan_to_num(heuristica_mat, nan=0, posinf=0, neginf=0)

    return heuristica_mat


def converter_para_array_corredor(corredor):
    """Converte um corredor para array numpy, independente do formato"""
    if isinstance(corredor, (list, tuple, np.ndarray)):
        return np.array(corredor)
    elif isinstance(corredor, dict):
        if 'coordenadas' in corredor:
            return np.array(corredor['coordenadas'])
        elif 'posicao' in corredor:
            return np.array(corredor['posicao'])
        elif 'x' in corredor and 'y' in corredor:
            return np.array([corredor['x'], corredor['y']])
        else:
            return np.array(list(corredor.values()))
    elif isinstance(corredor, (int, float)):
        return np.array([corredor])
    else:
        raise ValueError(f"Tipo de corredor não suportado: {type(corredor)}")


def calcular_linha_corredores(args):
    i, corredores = args
    A = corredores[i]
    linha = np.zeros(len(corredores))

    for j in range(len(corredores)):
        B = corredores[j]
        try:
            manhattan_dist = np.sum(np.abs(A - B))
            linha[j] = 1 / (1 + manhattan_dist)
        except Exception as e:
            linha[j] = 0 if i == j else 1

    return i, linha


def construir_matriz_heuristica_corredores_paralelo(dados):
    if not isinstance(dados, dict) or 'armazem' not in dados or 'corredores' not in dados['armazem']:
        raise ValueError("Estrutura de dados inválida. Deve conter 'armazem']['corredores'")

    try:
        list_corredores = [converter_para_array_corredor(corredor) for corredor in dados['armazem']['corredores']]
        dimensoes = [c.shape[0] for c in list_corredores]
        if len(set(dimensoes)) > 1:
            raise ValueError("Todos os corredores devem ter a mesma dimensionalidade")
        corredores = np.vstack(list_corredores)
    except Exception as e:
        raise ValueError(f"Falha ao converter corredores: {str(e)}")

    if len(corredores.shape) != 2:
        raise ValueError("Os corredores devem formar uma matriz 2D")

    quantidade_corredores = corredores.shape[0]
    heuristica_mat = np.zeros((quantidade_corredores, quantidade_corredores))

    num_processos = max(1, cpu_count() // 2)

    with Pool(processes=num_processos) as pool:
        args = [(i, corredores) for i in range(quantidade_corredores)]
        resultados = pool.map(calcular_linha_corredores, args)

    for i, linha in resultados:
        heuristica_mat[i, :] = linha

    np.fill_diagonal(heuristica_mat, 0)
    heuristica_mat = np.nan_to_num(heuristica_mat, nan=0, posinf=0, neginf=0)

    return heuristica_mat


def ant_colony_support(dados, iteracoes, formigas, evaporacao, feromonio, alfa, beta, heuristica_mat):
    feromonio_mat = np.ones(heuristica_mat.shape)
    melhor_solucao = None
    melhor_valor = 0
    iteracoes_sem_melhorar = 0
    for _ in range(iteracoes):
        ultimo_melhor_valor = melhor_valor
        if iteracoes_sem_melhorar > 20:
            break
        solucoes = []
        valores = []
        for formiga in range(formigas):
            dados['wave']['corredores'] = []
            solucao = construir_solucao_corredores(formiga, dados, feromonio_mat, heuristica_mat, alfa, beta)
            valor = funcao_objetivo(dados)
            solucoes.append(solucao)
            valores.append(valor)
            if valor > melhor_valor:
                melhor_valor = valor
                melhor_solucao = solucao
        if melhor_valor <= ultimo_melhor_valor:
            iteracoes_sem_melhorar += 1
        else:
            iteracoes_sem_melhorar = 0
        atualiza_feromonios(feromonio_mat, solucoes, valores, evaporacao, feromonio)
    return melhor_solucao, melhor_valor


def ant_colony3(dados, iteracoes, formigas_pedidos, formigas_corredores, evaporacao, feromonio, alfa, beta):
    print("Ant Colony 3")
    heuristica_mat_pedidos = construir_matriz_heuristica_pedidos_paralelo(dados)
    heuristica_mat_corredores = construir_matriz_heuristica_corredores_paralelo(dados)
    print('construiu heuristicas!')

    feromonio_mat_pedidos = np.ones(heuristica_mat_pedidos.shape, dtype=np.float64)

    melhor_solucao = None
    corredores_para_melhor_solucao = None
    melhor_valor = -np.inf
    iteracoes_sem_melhorar = 0

    num_processos = max(1, cpu_count() // 2)

    for iteracao in range(iteracoes):
        ultimo_melhor_valor = melhor_valor
        if iteracoes_sem_melhorar > 100:
            break

        args = [
            (formiga, dados, feromonio_mat_pedidos.copy(), heuristica_mat_pedidos, heuristica_mat_corredores,
             alfa, beta, formigas_corredores, evaporacao, feromonio)
            for formiga in range(formigas_pedidos)
        ]

        with Pool(processes=num_processos) as pool:
            resultados = pool.map(executar_formiga_pedidos, args)

        solucoes_pedidos, solucoes_corredores, valores = zip(*resultados)

        for sol_p, sol_c, valor in zip(solucoes_pedidos, solucoes_corredores, valores):
            if valor > melhor_valor:
                melhor_valor = valor
                melhor_solucao = sol_p
                corredores_para_melhor_solucao = sol_c
                iteracoes_sem_melhorar = 0

        if melhor_valor <= ultimo_melhor_valor:
            iteracoes_sem_melhorar += 1
        else:
            iteracoes_sem_melhorar = 0

        atualiza_feromonios(feromonio_mat_pedidos, solucoes_pedidos, valores, evaporacao, feromonio)

        print(f'Iteração {iteracao}: Melhor valor = {melhor_valor}')

    dados['wave']['pedidos'] = sorted(melhor_solucao)
    dados['wave']['corredores'] = sorted(corredores_para_melhor_solucao)

    print("Melhor valor encontrado:", melhor_valor)
    print("Pedidos selecionados:", dados['wave']['pedidos'])
    print("Corredores visitados:", dados['wave']['corredores'])

    return melhor_solucao, corredores_para_melhor_solucao, melhor_valor

def executar_formiga_pedidos(args):
    formiga, dados, feromonio_mat_pedidos, heuristica_mat_pedidos, heuristica_mat_corredores, alfa, beta, formigas_corredores, evaporacao, feromonio = args

    dados_local = dados.copy()
    dados_local['wave']['pedidos'] = []
    dados_local['wave']['tamanho'] = 0

    solucao_pedidos = construir_solucao_pedidos(
        formiga, dados_local, feromonio_mat_pedidos, heuristica_mat_pedidos, alfa, beta
    )

    dados_local['wave']['corredores'] = []
    solucao_corredores, valor = ant_colony_support(
        dados_local, 20, formigas_corredores, evaporacao, feromonio, alfa, beta, heuristica_mat_corredores
    )

    return solucao_pedidos, solucao_corredores, valor


def construir_solucao_pedidos(formiga, dados, feromonio_mat, heuristica_mat, alfa, beta):
    pedidos_keys = list(dados['armazem']['pedidos'].keys())
    atual = formiga
    caminho = [int(atual)]
    vertices_nao_visitados = {k for k in pedidos_keys if k != atual}
    while dados['wave']['tamanho'] + sum(dados['armazem']['pedidos'][atual]) > dados['wave']['limite_superior']:
        atual = np.random.choice(list(vertices_nao_visitados))
        vertices_nao_visitados.remove(atual)
        caminho = [int(atual)]
    dados['wave']['pedidos'] = [atual]
    dados['wave']['tamanho'] = sum(dados['armazem']['pedidos'][atual])

    while vertices_nao_visitados:
        vertices_list = list(vertices_nao_visitados)
        indices = [int(v) for v in vertices_list]

        feromonios = feromonio_mat[atual, indices]
        heuristicas = heuristica_mat[atual, indices]

        if np.any(np.isnan(feromonios)):
            raise ValueError(f"Feromônio contém NaN para o nó {atual} e índices {indices}")
        if np.any(np.isnan(heuristicas)):
            raise ValueError(f"Heurística contém NaN para o nó {atual} e índices {indices}")

        pontuacoes = (feromonios ** alfa) * (heuristicas ** beta)
        soma_pontuacoes = np.sum(pontuacoes)

        if soma_pontuacoes <= 0 or np.isnan(soma_pontuacoes):
            proximo = np.random.choice(vertices_list)
        else:
            prob = pontuacoes / soma_pontuacoes

            if np.any(np.isnan(prob)) or np.any(prob < 0) or not np.isclose(np.sum(prob), 1):
                proximo = np.random.choice(vertices_list)
            else:
                proximo = np.random.choice(vertices_list, p=prob)

        novo_tamanho = dados['wave']['tamanho'] + sum(dados['armazem']['pedidos'][proximo])
        if novo_tamanho > dados['wave']['limite_superior']:
            return caminho

        caminho.append(int(proximo))
        dados['wave']['pedidos'].append(proximo)
        dados['wave']['tamanho'] = novo_tamanho
        atual = proximo
        vertices_nao_visitados.remove(proximo)

    return caminho


def construir_solucao_corredores(formiga, dados, feromonio_mat, heuristica_mat, alfa, beta):
    pedidos_wave = dados['wave']['pedidos']
    pedidos_armazem = dados['armazem']['pedidos']

    # Converter demanda para float64 explicitamente
    demanda = np.sum([pedidos_armazem[pedido] for pedido in pedidos_wave], axis=0).astype(np.float64)
    oferta = np.zeros_like(demanda)  # Já será float64 por causa do dtype da demanda

    corredores_keys = list(dados['armazem']['corredores'].keys())
    # Garantir que os valores dos corredores são float64
    corredores_values = np.array(list(dados['armazem']['corredores'].values()), dtype=np.float64)

    atual = formiga
    caminho = [int(atual)]
    vertices_nao_visitados = {k for k in corredores_keys if k != atual}
    dados['wave']['corredores'] = [atual]

    while vertices_nao_visitados:
        oferta += corredores_values[atual]

        if np.all(demanda <= oferta):
            return caminho

        vertices_list = list(vertices_nao_visitados)
        indices = [int(v) for v in vertices_list]

        feromonios = feromonio_mat[atual, indices]
        heuristicas = heuristica_mat[atual, indices]
        pontuacoes = (feromonios ** alfa) * (heuristicas ** beta)

        prob = pontuacoes / np.sum(pontuacoes)
        proximo = np.random.choice(vertices_list, p=prob)

        caminho.append(int(proximo))
        dados['wave']['corredores'].append(proximo)
        atual = proximo
        vertices_nao_visitados.remove(proximo)

    return caminho

def atualiza_feromonios(feromonio_mat, solucoes, valores, evaporacao, feromonio, ferom_min=0.01):
    feromonio_mat *= (1 - evaporacao)
    np.maximum(feromonio_mat, ferom_min, out=feromonio_mat)

    melhores = np.argsort(valores)[-max(1, len(valores)//2):]
    for idx in melhores:
        solucao = solucoes[idx]
        valor = valores[idx]
        delta = feromonio / (1 + valor)
        for i in range(len(solucao) - 1):
            u, v = solucao[i], solucao[i + 1]
            feromonio_mat[u, v] += delta
            feromonio_mat[v, u] += delta