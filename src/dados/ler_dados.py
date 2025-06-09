def ler_dados(caminho, dados):
    # Inicialização da estrutura
    dados['wave'] = {
        'pedidos': [],
        'corredores': [],
        'limite_inferior': 0,
        'limite_superior': 0,
        'tamanho': 0
    }
    dados['armazem'] = {
        'pedidos': {},
        'corredores': {}
    }

    with open(caminho, 'r') as arquivo:
        linhas = [linha.strip() for linha in arquivo if linha.strip()]

        # Lê primeira linha (o, i, a)
        o, i, a = map(int, linhas[0].split())
        indice_linha = 1

        # Leitura dos pedidos
        for pedido_id in range(o):
            itens = list(map(int, linhas[indice_linha].split()))
            # Cria lista de tamanho i inicializada com 0
            lista_itens = [0] * i
            # Preenche os itens nas posições corretas
            for j in range(1, len(itens), 2):
                posicao = itens[j]
                valor = itens[j + 1]
                if posicao < i:  # Verificação de segurança
                    lista_itens[posicao] = valor
            dados['armazem']['pedidos'][pedido_id] = lista_itens
            indice_linha += 1

        # Leitura dos corredores
        for corredor_id in range(a):
            itens = list(map(int, linhas[indice_linha].split()))
            lista_itens = [0] * i
            for j in range(1, len(itens), 2):
                posicao = itens[j]
                valor = itens[j + 1]
                if posicao < i:
                    lista_itens[posicao] = valor
            dados['armazem']['corredores'][corredor_id] = lista_itens
            indice_linha += 1

        # Leitura dos limites
        limites = list(map(int, linhas[indice_linha].split()))
        dados['wave']['limite_inferior'] = limites[0]
        dados['wave']['limite_superior'] = limites[1]