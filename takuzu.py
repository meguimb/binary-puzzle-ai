# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 38:
# 99270 Margarida Bezerra
# 99272 Maria Sofia Mateus Pinho

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

def deepcopy_array(lst):
    lst_copy = np.zeros(len(lst), int)
    for i in range(len(lst)):
        lst_copy[i] = lst[i]
    return lst_copy

def get_half(N):
    if N%2==0:
        return N/2
    else:
        return N//2 + 1
class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

        N = self.board.N
        self.row_counter = np.zeros((N, 2), int)
        self.column_counter = np.zeros((N, 2), int)
        self.filled = 0
        for i in range(N):
            for j in range(N):
                curr = board.get_number(i, j)
                if (curr < 2):
                    self.row_counter[i][curr] += 1
                    self.column_counter[j][curr] += 1
                    self.filled += 1


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

    # verificar se há 2 0s de um lado e 2 1s do outro
    def check_impossible_play(self, i, j):
        N = self.board.N
        board = self.board
        if N < 5:
            return False
        if i < 2 and j < 2:
            return False
        if board.get_number(i-1,j)==board.get_number(i-2,j)==0 and board.get_number(i+1,j)==board.get_number(i+2,j)==1:
            return True
        if board.get_number(i-1,j)==board.get_number(i-2,j)==1 and board.get_number(i+1,j)==board.get_number(i+2,j)==0:
            return True
        if board.get_number(i,j-1)==board.get_number(i,j-2)==0 and board.get_number(i,j+1)==board.get_number(i,j+2)==1:
            return True
        if board.get_number(i,j-1)==board.get_number(i,j-2)==1 and board.get_number(i,j+1)==board.get_number(i,j+2)==0:
            return True
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
                if i == self.N-1 and j == self.N-1:
                    result += str(self.grid[i][j])
                elif(j == self.N-1):
                    result += str(self.grid[i][j]) + '\n'
                else:
                    result += str(self.grid[i][j]) + '\t'

        return result


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        init_state = TakuzuState(board)
        super().__init__(init_state)
        self.board = board
        self.curr_state = init_state


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

        # verificar se não completa 3 seguidos à direita e esquerda / cima e baixo
        if row >= 2 and (state.board.get_number(row-2, column) == state.board.get_number(row-1, column) == play):
            return False
        if row <= N-3 and (state.board.get_number(row+1, column) == state.board.get_number(row+2, column) == play):
            return False

        if column >= 2 and (state.board.get_number(row, column-2) == state.board.get_number(row, column-1) == play):
            return False
        if column <= N-3 and (state.board.get_number(row, column+1) == state.board.get_number(row, column+2) == play):
            return False

        return True 

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        added = False
        N = self.board.N
        for i in range(N):
            if state.row_counter[i][0] > get_half(N) or state.row_counter[i][1] > get_half(N):
                return []
            if state.column_counter[i][0] > get_half(N) or state.column_counter[i][1] > get_half(N):
                return []
            for j in range(N):
                if not self.is_free(i, j, state):
                    continue
                # check for imediate wrong plays already played
                else:
                    # check for compulsory plays
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
                    
                    # verificar se o número de 0s/1s numa linha/coluna obriga a uma jogada obrigatória
                    if (N % 2 == 0):
                        if (state.column_counter[j][0] == N/2):
                            return [(i, j, 1)]
                        elif (state.column_counter[j][1] == N/2):
                            return [(i, j, 0)]
                        elif (state.row_counter[i][0] == N/2):
                            return [(i, j, 1)]
                        elif (state.row_counter[i][1] == N/2):
                            return [(i, j, 0)]
                        elif (state.column_counter[j][0] > N/2):
                            return []
                        elif (state.column_counter[j][1] > N/2):
                            return []
                        elif (state.row_counter[i][0] > N/2):
                            return []
                        elif (state.row_counter[i][1] > N/2):
                            return []
                    elif (N % 2 == 1):
                        if (state.column_counter[j][0] == N/2+1):
                            return [(i, j, 1)]
                        elif (state.column_counter[j][1] == N/2+1):
                            return [(i, j, 0)]
                        elif (state.row_counter[i][0] == N/2+1):
                            return [(i, j, 1)]
                        elif (state.row_counter[i][1] == N/2+1):
                            return [(i, j, 0)]
                        elif (state.column_counter[j][0] > N/2+1):
                            return []
                        elif (state.column_counter[j][1] > N/2+1):
                            return []
                        elif (state.row_counter[i][0] > N/2+1):
                            return []
                        elif (state.row_counter[i][1] > N/2+1):
                            return []  
                    if state.check_impossible_play(i, j):
                        return []
                    if (added == False):
                        if self.can_add(i, j, 0, state):
                            actions += [(i, j, 0)]
                            added = True
                        if self.can_add(i, j, 1, state):
                            actions += [(i, j, 1)]
                            added = True
        # verificar se nenhuma das jogadas faz 2 linhas/2 colunas iguais
        if len(actions) == 2:
            row, column, play = actions[0]
            play2 = actions[1][2]
            grid_transposed = np.transpose(state.board.grid)
            row_cpy = deepcopy_array(state.board.grid[row])
            column_cpy = deepcopy_array(grid_transposed[column])
            row_cpy2 = deepcopy_array(state.board.grid[row])
            column_cpy2 = deepcopy_array(grid_transposed[column])
            row_cpy[column] = play
            column_cpy[row] = play
            row_cpy2[column] = play2
            column_cpy2[row] = play2
            for j in range(N):
                if np.array_equal(row_cpy, state.board.grid[j]):
                    return [actions[1]]
                if np.array_equal(column_cpy,grid_transposed[j]):
                    return [actions[1]]
                if np.array_equal(row_cpy2, state.board.grid[j]):
                    return [actions[0]]
                if np.array_equal(column_cpy2,grid_transposed[j]):
                    return [actions[0]]
        return actions


    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_state = state.do_action(action)
        self.curr_state = new_state
        return new_state

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        N = state.board.N

        # verificar se todos os espaços estão preenchidos
        if state.filled != (N*N):
            return False
        # verificar numero certo de 1s e 0s nas linhas e nas colunas
        for i in range(N):
            if (N % 2 == 0):
                if (state.row_counter[i][0] + state.row_counter[i][1] != N 
                    or state.row_counter[i][0] > N/2 or state.row_counter[i][1] > N/2): 
                    return False
                if (state.column_counter[i][0] + state.column_counter[i][1] != N
                    or state.column_counter[i][0] > N/2 or state.column_counter[i][1] > N/2): 
                    return False
            elif (N % 2 == 1):
                if (state.row_counter[i][0] + state.row_counter[i][1] != N 
                    or state.row_counter[i][0] > N/2+1 or state.row_counter[i][1] > N/2+1): 
                    return False
                if (state.column_counter[i][0] + state.column_counter[i][1] != N
                    or state.column_counter[i][0] > N/2+1 or state.column_counter[i][1] > N/2+1): 
                    return False

        # verificar se ha mais de dois 0s e 1s seguidos
        for i in range(N):
            for j in range(N-2):
                if (state.board.grid[i][j] == state.board.grid[i][j+1] == state.board.grid[i][j+2]):
                    return False
                if (state.board.grid[j][i] == state.board.grid[j+1][i] == state.board.grid[j+2][i]):
                    return False

        # verificar se ha duas linhas ou duas colunas iguais
        grid_transposed = np.transpose(state.board.grid)
        for i in range(N) :
            for j in range(i+1, N):
                if np.array_equal(state.board.grid[i], state.board.grid[j]):
                    return False
                if np.array_equal(grid_transposed[i],grid_transposed[j]):
                    return False
        return True
    

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        return self.curr_state.board.N - self.curr_state.filled

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    
    board = Board.parse_instance_from_stdin()
    problem = Takuzu(board)
    goal_node = astar_search(problem)
    print(goal_node.state.board)
