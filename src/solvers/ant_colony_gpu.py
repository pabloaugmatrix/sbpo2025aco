import cupy as cp
import numpy as np
from cupyx.profiler import benchmark
import time


class ACOOptimizado:
    def __init__(self, dados):
        self.dados = dados
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()

    def _inicializar_dados_gpu(self):
        """Inicializa os dados na GPU de forma otimizada e segura"""
        try:
            self.pedidos_gpu = cp.array(list(self.dados['armazem']['pedidos'].values()), dtype=cp.float32)
            self.corredores_gpu = cp.array(list(self.dados['armazem']['corredores'].values()), dtype=cp.float32)

            # Adiciona pequeno valor epsilon para evitar divisão por zero
            epsilon = 1e-10
            self.normas_pedidos = cp.linalg.norm(self.pedidos_gpu, axis=1) + epsilon
            self.normas_corredores = cp.linalg.norm(self.corredores_gpu, axis=1) + epsilon
        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar dados na GPU: {str(e)}")

    def _verificar_nan(self, array, nome="array"):
        """Verifica se há valores NaN no array"""
        if cp.any(cp.isnan(array)):
            raise ValueError(f"Valores NaN detectados em {nome}")

    def construir_matriz_heuristica_pedidos(self):
        """Calcula a matriz heurística com verificação de NaN"""
        try:
            produto_interno = cp.matmul(self.pedidos_gpu, self.pedidos_gpu.T)
            self._verificar_nan(produto_interno, "produto_interno")

            norma_matriz = cp.outer(self.normas_pedidos, self.normas_pedidos)
            self._verificar_nan(norma_matriz, "norma_matriz")

            heuristica = cp.zeros_like(produto_interno)
            mask = norma_matriz != 0
            heuristica[mask] = produto_interno[mask] / norma_matriz[mask]

            # Substitui quaisquer NaN restantes por 0
            heuristica = cp.nan_to_num(heuristica)
            return heuristica
        except Exception as e:
            raise RuntimeError(f"Erro ao construir matriz heurística de pedidos: {str(e)}")

    def construir_matriz_heuristica_corredores(self):
        """Calcula a matriz heurística com verificação de NaN"""
        try:
            produto_interno = cp.matmul(self.corredores_gpu, self.corredores_gpu.T)
            self._verificar_nan(produto_interno, "produto_interno")

            norma_matriz = cp.outer(self.normas_corredores, self.normas_corredores)
            self._verificar_nan(norma_matriz, "norma_matriz")

            heuristica = cp.ones_like(produto_interno)
            mask = norma_matriz != 0
            heuristica[mask] = 1 - (produto_interno[mask] / norma_matriz[mask])

            # Substitui quaisquer NaN restantes por 1
            heuristica = cp.nan_to_num(heuristica, nan=1.0)
            return heuristica
        except Exception as e:
            raise RuntimeError(f"Erro ao construir matriz heurística de corredores: {str(e)}")

    def _amostrar_proximo_vertice(self, probabilidades):
        """Método robusto para amostragem que trata NaN e zeros"""
        try:
            # Converter para numpy se necessário e garantir que é um vetor válido
            prob_np = cp.asnumpy(probabilidades)

            # Substituir NaN por 0 e normalizar
            prob_np = np.nan_to_num(prob_np)
            prob_np = np.maximum(prob_np, 0)  # Garante valores não-negativos

            # Se todos forem zero, usar distribuição uniforme
            if np.all(prob_np == 0):
                prob_np = np.ones_like(prob_np) / len(prob_np)
            else:
                prob_np = prob_np / np.sum(prob_np)  # Renormaliza

            return np.random.choice(len(prob_np), p=prob_np)
        except Exception as e:
            raise RuntimeError(f"Erro na amostragem do próximo vértice: {str(e)}")

    def construir_solucao_pedidos(self, feromonio_mat, heuristica_mat, alfa, beta):
        """Versão robusta da construção de solução"""
        try:
            atual = np.random.choice(list(self.dados['armazem']['pedidos'].keys()))
            caminho = [int(atual)]
            vertices_nao_visitados = [i for i in range(len(self.dados['armazem']['pedidos'])) if i != atual]

            feromonio_gpu = cp.asarray(feromonio_mat, dtype=cp.float32)
            heuristica_gpu = cp.asarray(heuristica_mat, dtype=cp.float32)

            while vertices_nao_visitados:
                vertices_gpu = cp.array(vertices_nao_visitados)
                feromonios = feromonio_gpu[atual, vertices_gpu]
                heuristicas = heuristica_gpu[atual, vertices_gpu]

                # Cálculo seguro das pontuações
                pontuacoes = (feromonios ** alfa) * (heuristicas ** beta)
                pontuacoes = cp.nan_to_num(pontuacoes, nan=0.0, posinf=1e10, neginf=0.0)

                # Evitar divisão por zero
                soma_pontuacoes = cp.sum(pontuacoes)
                if soma_pontuacoes == 0:
                    probabilidades = cp.ones_like(pontuacoes) / len(pontuacoes)
                else:
                    probabilidades = pontuacoes / soma_pontuacoes

                indice = self._amostrar_proximo_vertice(probabilidades)
                proximo = vertices_nao_visitados[indice]

                # Verificação de capacidade
                if (sum(self.dados['armazem']['pedidos'][proximo]) + self.dados['wave']['tamanho']) > \
                        self.dados['wave']['limite_superior']:
                    return caminho

                caminho.append(proximo)
                self.dados['wave']['pedidos'].append(proximo)
                self.dados['wave']['tamanho'] += sum(self.dados['armazem']['pedidos'][proximo])
                atual = proximo
                vertices_nao_visitados.remove(proximo)

            return caminho
        except Exception as e:
            raise RuntimeError(f"Erro ao construir solução de pedidos: {str(e)}")

    # ... (mantenha o restante dos métodos como no código anterior)

    def executar_colonia_formigas(self, iteracoes, formigas, evaporacao, feromonio, alfa, beta):
        """Versão robusta do algoritmo principal"""
        try:
            self._inicializar_dados_gpu()

            # Construir matrizes com verificação de erros
            heuristica_pedidos = self.construir_matriz_heuristica_pedidos()
            heuristica_corredores = self.construir_matriz_heuristica_corredores()

            feromonio_pedidos = cp.ones_like(heuristica_pedidos)
            melhor_solucao = None
            melhor_corredores = None
            melhor_valor = -np.inf

            for iteracao in range(iteracoes):
                solucoes = []
                valores = []

                for _ in range(formigas):
                    try:
                        self.dados['wave']['pedidos'] = []
                        self.dados['wave']['tamanho'] = 0
                        self.dados['wave']['corredores'] = []

                        solucao = self.construir_solucao_pedidos(
                            feromonio_pedidos.get(),
                            heuristica_pedidos.get(),
                            alfa, beta
                        )

                        corredores = self.construir_solucao_corredores(
                            feromonio_pedidos.get(),
                            heuristica_corredores.get(),
                            alfa, beta
                        )

                        valor = self.funcao_objetivo()

                        if valor > melhor_valor:
                            melhor_valor = valor
                            melhor_solucao = solucao
                            melhor_corredores = corredores

                        solucoes.append(solucao)
                        valores.append(valor)
                    except Exception as e:
                        print(f"Erro na formiga {_}: {str(e)}")
                        continue

                # Atualização segura dos feromônios
                try:
                    feromonio_pedidos = cp.asarray(
                        self.atualizar_feromonios(
                            feromonio_pedidos.get(),
                            solucoes,
                            valores,
                            evaporacao,
                            feromonio
                        ),
                        dtype=cp.float32
                    )
                except Exception as e:
                    print(f"Erro ao atualizar feromônios: {str(e)}")
                    continue

                if iteracao % 10 == 0:
                    self.memory_pool.free_all_blocks()
                    self.pinned_memory_pool.free_all_blocks()

            return melhor_solucao, melhor_corredores, melhor_valor
        except Exception as e:
            raise RuntimeError(f"Erro na execução da colônia de formigas: {str(e)}")

    def funcao_objetivo(self):
        return self.dados['wave']['tamanho']/len(self.dados['wave']['corredores'])