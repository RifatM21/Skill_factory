import numpy as np


def find_num(number, predict, left=1, right=100):
    global count
    while predict != number:
        count += 1
        if right - left == 2:
            predict = (left + right) // 2
        elif predict > number:
            if right > predict:
                right = predict
            else:
                predict = right
            predict -= ((left + right) // 2)
            return find_num(number, predict, left, right)
        else:
            if left < predict:
                left = predict
            else:
                predict = left
            predict += ((right - left) // 2) + 1
            return find_num(number, predict, left, right)
    return count



def game_core_v3(number):
    global count
    count = 1
    predict = np.random.randint(1, 101)
    find_num(number, predict)
    return count


def score_game(game_core_v3):
    count_ls = []
    np.random.seed(1)
    random_array = np.random.randint(1, 101, size=1000)
    for number in random_array:
        count_ls.append(game_core_v3(number))
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return score


score_game(game_core_v3)
