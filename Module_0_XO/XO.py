board = []
for i in range(9):
    board.append('-')


def create_board(board):
    print(' ', '0', '1', '2')
    for i in range(3):
        print(i, board[0 + i * 3], board[1 + i * 3], board[2 + i * 3])


def move(players_sign):
    move_ability = False
    while not move_ability:
        players_answer = list(map(int, input('Координаты поля, куда поставим ' + players_sign + ': ').split()))
        if -1 < (players_answer[0] and players_answer[1]) < 3:
            if str(board[players_answer[1] * 3 + players_answer[0]]) not in "x0":
                board[players_answer[1] * 3 + players_answer[0]] = players_sign
                move_ability = True
            else:
                print('Это поле уже занято')
        else:
            print('Некорректные координаты. Координаты должны быть от 0 до 2')


def win_check(board):
    win_coombinations = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
    for coombination in win_coombinations:
        if board[coombination[0]] == board[coombination[1]] == board[coombination[2]]:
            return board[coombination[0]]
    return False


def game(board):
    count = 0
    win = False
    while not win:
        create_board(board)
        if count % 2 == 0:
            move('x')
        else:
            move('0')
        count += 1
        if count > 4:
            temp = win_check(board)
            if temp:
                print('Выиграл ', temp, '!', sep='')
                win = True
                break
        if count == 9:
            print('Ничья')
            break
    create_board(board)


game(board)







