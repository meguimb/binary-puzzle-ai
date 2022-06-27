# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

from pickle import FALSE
import sys
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

        N = self.board.N
        self.row_counter = np.zeros((N, 2), int)
        self.column_counter = np.zeros((N, 2), int)
        
        for i in range(N):
            for j in range(N):
                curr = board.get_number(i, j)
                if (curr < 2):
                    self.row_counter[i][curr] += 1
                    self.column_counter[j][curr] += 1

    def __lt__(self, other):
        return self.id < other.id

    def do_action(self, action):
        row, column, num = action
        if (self.board.grid[row][column] == 2):
            next_grid = np.array([row[:] for row in self.board.grid])
            next_grid[row][column] = num
            next_board = Board(self.board.N, next_grid)
            next_state = TakuzuState(next_board)
            return next_state
        return None
    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, N, board):
        self.N = N
        self.grid = board

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.grid[row][col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        if(row >= self.N-1):
            below = None
        else:
            below = self.get_number(row+1, col)
        
        if(row <= 0):
            above = None
        else:
            above = self.get_number(row-1, col)

        return (below, above)
        

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if(col <= 0):
            left = None
        else:
            left = self.get_number(row, col-1)
        
        if(col >= self.N-1):
            right = None
        else:
            right = self.get_number(row, col+1)

        return (left, right)

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        # add error checking
        lines = []
        # receber linhas do ficheiro de input, incluindo N
        for line in sys.stdin:
            lines = lines + [line]
        N = int(lines[0])

        # criar lista vazia NxN
        grid = np.empty((N, N), int)
        
        # criar uma array 2d e pôr os valores do ficheiro de input
        for i in range(N):
            line_split = lines[i+1].split("\t")
            for j in range(N):
                grid[i][j] = int(line_split[j])
        
        return Board(N, grid)

    # TODO: outros metodos da classe
    def __str__(self):
        result = ''
        for i in range(self.N):
            for j in range(self.N):
                if(j == self.N-1):
                    result += str(self.grid[i][j]) + '\n'
                else:
                    result += str(self.grid[i][j]) + '\t'

        return result


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        """O construtor especifica o estado inicial."""
        super().__init__(TakuzuState(board))
        self.board = board
        N = board.N
        self.row_counter = np.zeros((N, 2), int)
        self.column_counter = np.zeros((N, 2), int)
        # tomando como ex a instancia de takuzu na pagina 2
        # column 0's and 1's: [[0, 1], [1, 2], [1, 0], [2, 0]] => na 1a coluna tem 0 "0"s e 1 "1"
        # row 0's and 1's: [[1, 1], [1, 0], [1, 0], [1, 2]]
        
        for i in range(N):
            for j in range(N):
                curr = board.get_number(i, j)
                if (curr < 2):
                    self.row_counter[i][curr] += 1
                    self.column_counter[j][curr] += 1

    def is_free(self, row, column, state:TakuzuState):
        return (state.board.get_number(row, column) == 2 )

    def can_add(self, row, column, play, state: TakuzuState):
        N = state.board.N
        if not self.is_free(row, column, state):
            return False

        # if (state.board.get_number(row, column)) != 2:
        #    return False
        if (state.row_counter[row][play] > round(state.board.N/2)):
            return False
        if (state.board.adjacent_horizontal_numbers(row, column) == (play, play)):
            return False
        if (state.board.adjacent_vertical_numbers(row, column) == (play, play)):
            return False
        if (N % 2 == 0):
            if (state.row_counter[row][play] >= N/2 or 
                state.column_counter[column][play] >= N/2):
                return False
        if (N % 2 == 1):
            if (state.row_counter[row][play] >= N/2+1 or 
                state.column_counter[column][play] >= N/2+1):
                return False
        # verificar se não compelta 3 seguidos à direita e esquerda / cima e baixo
        return True 

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        added = False
        N = self.board.N
        for i in range(N):
            for j in range(N):
                if not self.is_free(i, j, state):
                    continue
                else:
                    if (state.board.adjacent_horizontal_numbers(i, j)[0] == state.board.adjacent_horizontal_numbers(i, j)[1] != 2):
                        return [(i, j, 1 - state.board.adjacent_horizontal_numbers(i, j)[0])]
                    if (state.board.adjacent_vertical_numbers(i, j)[0] == state.board.adjacent_vertical_numbers(i, j)[1] != 2):
                        return [(i, j, 1 - state.board.adjacent_vertical_numbers(i, j)[0])]

                    if (i >= 2 and (state.board.get_number(i-2, j) == state.board.get_number(i-1, j) != 2)):
                        return [(i, j, 1 - state.board.get_number(i-1, j))]
                    if (i <= N-3 and (state.board.get_number(i+1, j) == state.board.get_number(i+2, j) != 2)):
                        return [(i, j, 1 - state.board.get_number(i+1, j))]

                    if (j >= 2 and (state.board.get_number(i, j-2) == state.board.get_number(i, j-1) != 2)):
                        return [(i, j, 1 - state.board.get_number(i, j-1))]
                    if (j <= N-3 and (state.board.get_number(i, j+1) == state.board.get_number(i, j+2) != 2)):
                        return [(i, j, 1 - state.board.get_number(i, j+1))]

                    if (added == False):
                        if self.can_add(i, j, 0, state):
                            actions += [(i, j, 0)]
                            added = True
                        if self.can_add(i, j, 1, state):
                            actions += [(i, j, 1)]
                            added = True
        return actions


    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_state = state.do_action(action)
        return new_state

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        N = state.board.N

        # apagar:
        full = True
        for i in range(N):
            for j in range(N):
                if state.board.get_number(i, j) == 2:
                    full = False
        if (full == True):
            print(state.board)

        # verificar numero certo de 1s e 0s nas linhas e nas colunas
        for i in range(N):
            for j in range(N):
                if state.board.get_number(i, j) == 2:
                    print('false1')
                    return False
            if (N % 2 == 0):
                if (state.row_counter[i][0] + state.row_counter[i][1] != N 
                    or state.row_counter[i][0] > N/2 or state.row_counter[i][1] > N/2): 
                    print('false2')
                    return False
                if (state.column_counter[i][0] + state.column_counter[i][1] != N
                    or state.column_counter[i][0] > N/2 or state.column_counter[i][1] > N/2): 
                    print('false3')
                    return False
            elif (N % 2 == 1):
                if (state.row_counter[i][0] + state.row_counter[i][1] != N 
                    or state.row_counter[i][0] > N/2+1 or state.row_counter[i][1] > N/2+1): 
                    print('false2')
                    return False
                if (state.column_counter[i][0] + state.column_counter[i][1] != N
                    or state.column_counter[i][0] > N/2+1 or state.column_counter[i][1] > N/2+1): 
                    print('false3')
                    return False

        # verificar se ha mais de dois 0s e 1s seguidos
        for i in range(N):
            for j in range(N-2):
                if (state.board.grid[i][j] == state.board.grid[i][j+1] == state.board.grid[i][j+2]):
                    print('false4')
                    return False
        for i in range(N-2):
            for j in range(N):
                if (state.board.grid[i][j] == state.board.grid[i+1][j] == state.board.grid[i+2][j]):
                    print('false5')
                    return False

        # verificar se ha duas linhas ou duas colunas iguais
        grid_transposed = np.transpose(state.board.grid)
        for i in range(N) :
            for j in range(i+1, N):
                if np.array_equal(state.board.grid[i], state.board.grid[j]):
                    print('false6')
                    return False
                if np.array_equal(grid_transposed[i],grid_transposed[j]):
                    print('false7')
                    return False
        print('true')
        return True
    

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    
    board = Board.parse_instance_from_stdin()
    problem = Takuzu(board)
    goal_node = depth_first_tree_search(problem)
    print(goal_node.state.board)
