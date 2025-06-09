import re
import sys
from pathlib import Path

import pandas as pd


def ler_arquivo_resultados(caminho_txt):
    # Expressão regular para extrair os dados da linha
    padrao = re.compile(
        r'Execução (\d+), seed: (\d+), func_obj: ([\d.]+), tempo: ([\d.]+) segundos'
    )

    dados = []

    with open(caminho_txt, 'r') as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            match = padrao.match(linha)
            if match:
                execucao = int(match.group(1))
                seed = int(match.group(2))
                func_obj = float(match.group(3))
                tempo = float(match.group(4))
                dados.append((execucao, seed, func_obj, tempo))

    # Criar DataFrame para facilitar análises
    df = pd.DataFrame(dados, columns=['execucao', 'seed', 'func_obj', 'tempo'])
    return df

if __name__ == '__main__':
    # Exemplo de uso:
    instancia = sys.argv[1]
    caminho = Path(f'data/benchmark/{instancia}_benchmark.txt')
    df_resultados = ler_arquivo_resultados(caminho)

    # Exibir os dados
    print(df_resultados)

    #  calculos:
    print("\nMédia da função objetivo por seed:")
    print(df_resultados.groupby('seed')['func_obj'].mean())
    print('\nModa da função objetivo (todas as seeds):')
    moda_global = df_resultados['func_obj'].mode()
    print(', '.join(map(str, moda_global.values)))

    print("\nTempo médio por seed:")
    print(df_resultados.groupby('seed')['tempo'].mean())

    print('\nTempo medio total:')
    print(df_resultados.groupby('seed')['tempo'].mean().mean())

    print("\nMelhor resultado geral:")
    print(df_resultados.loc[df_resultados['func_obj'].idxmin()])


