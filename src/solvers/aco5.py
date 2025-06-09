from sklearn.metrics.pairwise import manhattan_distances
from multiprocessing import Pool, cpu_count
import random
import numpy as np
import concurrent.futures
import copy

from src.funcao_objetivo.funcao_objetivo import funcao_objetivo

import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from multiprocessing import Pool, cpu_count
from functools import partial


def calcular_bloco_heuristica(args):
    """
    Calcula um bloco da matriz heurística para melhor aproveitamento de cache
    e redução de overhead na comunicação entre processos.
    """
    start_i, end_i, start_j, end_j, dados, quantidade_pedidos, dimensao = args
    bloco = np.zeros((end_i - start_i, end_j - start_j))

    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            if i >= quantidade_pedidos and j >= quantidade_pedidos:
                # Otimização: pré-calcula índices
                idx_i = i - quantidade_pedidos
                idx_j = j - quantidade_pedidos
                bloco[i - start_i, j - start_j] = manhattan_distances(
                    [dados['armazem']['corredores'][idx_i]],
                    [dados['armazem']['corredores'][idx_j]])[0][0]
            elif i < quantidade_pedidos and j < quantidade_pedidos:
                bloco[i - start_i, j - start_j] = manhattan_distances(
                    [dados['armazem']['pedidos'][i]],
                    [dados['armazem']['pedidos'][j]])[0][0]
            elif i < quantidade_pedidos and j >= quantidade_pedidos:
                bloco[i - start_i, j - start_j] = manhattan_distances(
                    [dados['armazem']['pedidos'][i]],
                    [dados['armazem']['corredores'][j - quantidade_pedidos]])[0][0]
            else:
                bloco[i - start_i, j - start_j] = manhattan_distances(
                    [dados['armazem']['corredores'][i - quantidade_pedidos]],
                    [dados['armazem']['pedidos'][j]])[0][0]

    return start_i, end_i, start_j, end_j, 1 / (bloco + 1e-10)


def construir_heuristica(dados, block_size=50):
    """
    Versão altamente otimizada com:
    - Paralelismo por blocos (melhor para cache)
    - Pré-processamento de dados
    - Balanceamento dinâmico de carga
    """
    # Validação de entrada acelerada
    try:
        quantidade_pedidos = len(dados['armazem']['pedidos'])
        quantidade_corredores = len(dados['armazem']['corredores'])
    except (KeyError, TypeError) as e:
        raise ValueError("Estrutura de dados inválida") from e

    dimensao = quantidade_pedidos + quantidade_corredores
    heuristica = np.zeros((dimensao, dimensao))

    # Pré-processamento para reduzir acesso a dict
    corredores = np.array(dados['armazem']['corredores'])
    pedidos = np.array(dados['armazem']['pedidos'])

    # Otimização: tamanho do bloco baseado na dimensão
    block_size = min(block_size, dimensao // (cpu_count() * 2) or 1)

    # Gerar blocos para processamento paralelo
    blocks = []
    for i in range(0, dimensao, block_size):
        for j in range(0, dimensao, block_size):
            blocks.append((
                i, min(i + block_size, dimensao),
                j, min(j + block_size, dimensao),
                dados, quantidade_pedidos, dimensao
            ))

    # Processamento paralelo com chunksize ajustado
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(calcular_bloco_heuristica, blocks,
                           chunksize=max(1, len(blocks) // (cpu_count() * 4)))

    # Montagem da matriz final
    for start_i, end_i, start_j, end_j, bloco in results:
        heuristica[start_i:end_i, start_j:end_j] = bloco

    return heuristica


def demanda_atendida(demanda, oferta):
    return np.all(demanda <= oferta)

def construir_solucao(dados, formiga, heuristica_mat, feromonio_mat, alfa, beta):
    pedidos_range = len(dados['armazem']['pedidos'])
    corredores_range = len(dados['armazem']['corredores'])
    total_vertices = pedidos_range + corredores_range

    pedidos_disponiveis = [i for i in range(pedidos_range)]
    corredores_disponiveis = [i for i in range(pedidos_range, total_vertices)]
    vertices_restantes = [i for i in range(total_vertices)]

    caminho = []

    # Inicialização de demanda e oferta
    demanda = np.zeros(len(dados['armazem']['pedidos'][0]))
    oferta = np.zeros(len(dados['armazem']['corredores'][0]))

    pedidos_no_caminho = 0
    corredores_no_caminho = 0

    # Seleção inicial
    atual = random.choice(list(vertices_restantes))
    vertices_restantes.remove(atual)

    if atual < pedidos_range:
        pedidos_disponiveis.remove(atual)

        while sum(dados['armazem']['pedidos'][atual]) > dados['wave']['limite_superior']:
            if not pedidos_disponiveis:
                break  # Não há mais pedidos viáveis
            atual = random.choice(list(pedidos_disponiveis))
            pedidos_disponiveis.remove(atual)
            vertices_restantes.remove(atual)

        demanda += np.array(dados['armazem']['pedidos'][atual])
        pedidos_no_caminho += 1

    else:
        corredores_disponiveis.remove(atual)
        oferta += np.array(dados['armazem']['corredores'][atual - pedidos_range])
        corredores_no_caminho += 1

    caminho.append(atual)


    while ((not demanda_atendida(demanda, oferta) or pedidos_no_caminho == 0 or corredores_no_caminho == 0)) and vertices_restantes:

        pontuacoes = []


        for vertice in vertices_restantes:
            feromonio = feromonio_mat[atual, vertice]
            heuristica = heuristica_mat[atual, vertice]
            pontuacao = (feromonio ** alfa) * (heuristica ** beta)
            pontuacoes.append(pontuacao)

        pontuacoes = np.array(pontuacoes)

        if np.sum(pontuacoes) == 0:
            break  # Não há caminho viável

        probabilidades = pontuacoes / np.sum(pontuacoes)
        prob_cumsum = np.cumsum(probabilidades)
        r = np.random.random()
        indice = int(np.searchsorted(prob_cumsum, r))

        proximo = vertices_restantes[indice]
        vertices_restantes.remove(proximo)

        if proximo < pedidos_range:
            pedido = dados['armazem']['pedidos'][proximo]
            if np.sum(demanda) + sum(pedido) <= dados['wave']['limite_superior']:
                demanda += np.array(pedido)
                pedidos_no_caminho += 1
                pedidos_disponiveis.remove(proximo)
                caminho.append(proximo)

                if np.sum(demanda) == dados['wave']['limite_superior']:
                    vertices_restantes = [v for v in vertices_restantes if v not in pedidos_disponiveis]
            else:
                continue

        else:
            corredor = dados['armazem']['corredores'][proximo - pedidos_range]
            oferta += np.array(corredor)
            corredores_no_caminho += 1
            corredores_disponiveis.remove(proximo)
            caminho.append(proximo)

    dados['wave']['pedidos'] = [v for v in caminho if v < pedidos_range]
    dados['wave']['corredores'] = [v - pedidos_range for v in caminho if v >= pedidos_range]
    dados['wave']['tamanho'] = sum(sum(dados['armazem']['pedidos'][i]) for i in dados['wave']['pedidos'])

    return caminho


def atualiza_feromonios(feromonio_mat, solucoes, valores, evaporacao, feromonio, ferom_min=1):
    # Aplica evaporação
    feromonio_mat *= 1 - evaporacao
    # Garante um valor mínimo de feromônio
    feromonio_mat[feromonio_mat < ferom_min] = ferom_min
    for solucao, valor in zip(solucoes, valores):
        if valor <= 0:
            continue
        delta = feromonio * valor
        for i in range(len(solucao) - 1):
            # Verifica índices válidos
            if (0 <= solucao[i] < feromonio_mat.shape[0] and
                    0 <= solucao[i + 1] < feromonio_mat.shape[1]):
                feromonio_mat[solucao[i], solucao[i + 1]] += delta
                #feromonio_mat[solucao[i + 1], solucao[i]] += delta

def reseta_feromonios(feromonio_mat, feromonio):
    feromonio_mat = np.ones(feromonio_mat.shape) * feromonio


def ant_colony(dados, iteracoes, formigas, evaporacao, feromonio, alfa, beta, seed):
    random.seed(seed)
    np.random.seed(seed)
    #print("Ant Colony")
    heuristica_mat = construir_heuristica(dados)
    #print("Construiu heuristica")
    feromonio_mat = np.ones(heuristica_mat.shape) * feromonio
    #print("Construiu feromonio")

    melhores_pedidos = []
    melhores_corredores = []
    melhor_tamanho = 0
    melhor_valor = 0
    iteracoes_sem_melhorar = 0

    for iteracao in range(iteracoes):
        ultimo_melhor_valor = melhor_valor
        if iteracoes_sem_melhorar >= 100:
            break

        solucoes = []
        valores = []

        def tarefa(formiga_id):
            dados_local = copy.deepcopy(dados)  # cópia isolada
            dados_local['wave']['pedidos'] = []
            dados_local['wave']['corredores'] = []
            dados_local['wave']['tamanho'] = 0

            solucao = construir_solucao(dados_local, formiga_id, heuristica_mat, feromonio_mat, alfa, beta)
            valor = funcao_objetivo(dados_local)
            return solucao, valor, dados_local['wave']['pedidos'], dados_local['wave']['corredores'], dados_local['wave']['tamanho']

        with concurrent.futures.ThreadPoolExecutor() as executor:
            resultados = list(executor.map(tarefa, range(formigas)))

        for solucao, valor, pedidos, corredores, tamanho in resultados:
            solucoes.append(solucao)
            valores.append(valor)
            if valor > melhor_valor:
                melhor_valor = valor
                melhores_pedidos = pedidos
                melhores_corredores = corredores
                melhor_tamanho = tamanho



        if melhor_valor <= ultimo_melhor_valor:
            iteracoes_sem_melhorar += 1
        else:
            iteracoes_sem_melhorar = 0
        #if iteracoes_sem_melhorar > 45:
            #reseta_feromonios(feromonio_mat, feromonio)
            #pass

        atualiza_feromonios(feromonio_mat, solucoes, valores, evaporacao, feromonio)
        #print(f'Iteracao {iteracao}, melhor valor {melhor_valor}')
        #print(f'Melhor valor da iteracao: {max(valores)}')

    dados['wave']['pedidos'] = melhores_pedidos
    dados['wave']['corredores'] = melhores_corredores
    dados['wave']['tamanho'] = melhor_tamanho

    #print(dados['wave']['pedidos'])
    #print(dados['wave']['corredores'])
    #print(funcao_objetivo(dados))
