from src.funcao_objetivo.funcao_objetivo import funcao_objetivo


def mockup_instance_0020(dados):
    dados['wave']['pedidos'] = [0, 1, 2, 4]
    tamanho = 0
    for pedido in dados['wave']['pedidos']:
        tamanho += sum(dados['armazem']['pedidos'][pedido])
    dados['wave']['tamanho'] = tamanho
    dados['wave']['corredores'] = [1, 3]
    print(f'Valor do Mockup: 0020 - {funcao_objetivo(dados)}')