import sys
from pathlib import Path

from src.dados.escrever_solucao import escrever_solucao
from src.dados.ler_dados import ler_dados
from src.funcao_objetivo.funcao_objetivo import funcao_objetivo
from src.mockup.mockup_instance_0020 import mockup_instance_0020

from src.solvers.aco5 import ant_colony


import time
import random

import time
import random
import multiprocessing
import os

import time
import random
import multiprocessing
import os

import time
import random
import multiprocessing
import os
import signal
import psutil

def run_ant_colony(dados, formigas, result_queue, seed):
    try:
        ant_colony(dados, 1000, formigas, 0.5, 1, 1, 2, seed)
        valor = funcao_objetivo(dados)
        result_queue.put(valor)
    except Exception as e:
        result_queue.put(f"error: {e}")

def terminate_process_tree(pid):
    """Finaliza processo e todos os subprocessos"""
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def benchmark_ant_colony(instance_name, dados, formigas):
    multiprocessing.set_start_method("spawn", force=True)
    os.makedirs("data/benchmark", exist_ok=True)
    output_file = f"data/benchmark/{instance_name}_benchmark.txt"

    with open(output_file, "w") as f:
        for seed in range(5):
            for i in range(5):
                print(f"instance: {instance_name}")
                print(f"seed: {seed}")
                print(f"iteration: {i}")
                random.seed(seed)
                start_time = time.time()

                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(target=run_ant_colony, args=(dados, formigas, result_queue, seed))
                process.start()
                process.join(timeout=600)

                end_time = time.time()
                elapsed_time = end_time - start_time

                if process.is_alive():
                    terminate_process_tree(process.pid)
                    process.join()
                    f.write(f"Execução {i}, seed: {seed}, func_obj: timeout, tempo: timeout\n")
                    print(f"Timeout detectado. Benchmark interrompido. Resultados parciais salvos em '{output_file}'.")
                    return

                valor = result_queue.get() if not result_queue.empty() else "error"
                f.write(f"Execução {i}, seed: {seed}, func_obj: {valor}, tempo: {elapsed_time:.6f} segundos\n")

    print(f"Benchmark finalizado. Resultados salvos em '{output_file}'.")





if __name__ == "__main__":
    instancia = sys.argv[1]
    caminho_entrada = Path('data/input/') / instancia
    dados = {}
    ler_dados(caminho_entrada, dados)
    formigas = int((len(dados['armazem']['pedidos']) + len(dados['armazem']['corredores']))/2)
    if formigas < 1:
        formigas = 1
    benchmark_ant_colony(instancia, dados, formigas)
    #ant_colony(dados, 1000, formigas, 0.5, 1, 1, 2)


    mockup_instance_0020(dados)
    caminho_saida = Path('data/output/') / instancia
    escrever_solucao(caminho_saida, dados)


