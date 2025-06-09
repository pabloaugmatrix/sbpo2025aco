

def escrever_solucao(caminho, dados):
    with open(caminho, 'w') as arquivo:
        arquivo.write(f'{len(dados["wave"]["pedidos"])}\n')
        for pedido in dados["wave"]["pedidos"]:
            arquivo.write(f'{pedido}\n')
        arquivo.write(f'{len(dados["wave"]["corredores"])}\n')
        for corredor in dados["wave"]["corredores"]:
            arquivo.write(f'{corredor}\n')
