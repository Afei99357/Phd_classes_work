## game 1
game_1_1 = "1352361135421"
game_1_2 = "654663646462"
game_1_3 = "3545431523"
game_1_4 = "6162566366"
game_1_5 = "24125615436"
game_1_6 = "156661263"
game_1_7 = "416251"
game_1_8 = "2646612"
game_1_9 = "5126514253451"
game_1_10 = "646435166465"
game_1_11 = "1235414612541241"
game_1_12 = "463636"

## game 2
game_2_1 = "654563646462"
game_2_2 = "1352364125421"
game_2_3 = "6162556366"
game_2_4 = "354543152324125615436466561"
game_2_5 = "1536612632646612"
game_2_6 = "5124564231451"
game_2_7 = "646435166465"
game_2_8 = "12654146"
game_2_9 = "463636"
game_2_10 = "12541266"

## game 3
game_3_1 = "463636"
game_3_2 = "1352366125421"
game_3_3 = "654563646462"
game_3_4 = "3545431523"
game_3_5 = "6162556366"
game_3_6 = "24125615436"
game_3_7 = "143661263"
game_3_8 = "466561"
game_3_9 = "6122646"
game_3_10 = "5126514253351"
game_3_11 = "646436465166"
game_3_12 = "1265414612541266"

## game 4
game_4_1 = "153661666264"
game_4_2 = "415436453521"
game_4_3 = "666162556366"
game_4_4 = "35454315232412"
game_4_5 = "241636666"
game_4_6 = "1213523161254212"
game_4_7 = "612646435166"
game_4_8 = "51235142525151265414612541266"
game_4_9 = "465463636"

# game_1 = game_1_1+game_1_2+game_1_3+game_1_4+game_1_5+game_1_6+game_1_7+game_1_8+game_1_9+game_1_10+game_1_11+game_1_12
game_fair = game_1_1 + game_1_3 + game_1_5 + game_1_7 + game_1_9 + game_1_11 + game_2_2 + game_2_4 + game_2_6 + game_2_8 + \
            game_2_10 + game_3_2 + game_3_4 + game_3_6 + game_3_8 + game_3_10 + game_3_12 + game_4_2 + game_4_4 + game_4_6 + game_4_8

game_loaded = game_1_2 + game_1_4 + game_1_6 + game_1_8 + game_1_10 + game_1_12 + game_2_1 + game_2_3 + game_2_5 + game_2_7 \
              + game_2_9 + game_3_1 + game_3_3 + game_3_5 + game_3_7 + game_3_9 + game_3_11 + game_4_1 + game_4_3 + game_4_5 + game_4_7 + game_4_9

game_fair_len = len(game_fair)
game_loaded_len = len(game_loaded)

A01 = 19
A10 = 20
N01 = game_fair_len
N10 = game_loaded_len

# print(len(game_1))
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
count_6 = 0
for i in game_fair:
    if i == "1":
        count_1 = count_1 + 1
    if i == "2":
        count_2 = count_2 + 1
    if i == "3":
        count_3 = count_3 + 1
    if i == "4":
        count_4 = count_4 + 1
    if i == "5":
        count_5 = count_5 + 1
    if i == "6":
        count_6 = count_6 + 1

e01_1 = count_1 / N01
e02_1 = count_2 / N01
e03_1 = count_3 / N01
e04_1 = count_4 / N01
e05_1 = count_5 / N01
e06_1 = count_6 / N01

print("Start probability for fair:  " + str(1 / 4) + '\n' + "End probability for loaded: " + str(2 / game_fair_len))
print("Transition probability for fair to loaded: " + str(A01 / N01))
print("Emission Probability for fair to loaded: " + str(e01_1) + ", " + str(e02_1) + ", " + str(e03_1) + ", "
      + str(e04_1) + ", " + str(e05_1) + ", " + str(e06_1))
print("Transition probability from fair to fair " + str(1- A01 / N01 - 2 / game_fair_len) + "\n")

count_7 = 0
count_8 = 0
count_9 = 0
count_10 = 0
count_11 = 0
count_12 = 0
for i in game_loaded:
    if i == "1":
        count_7 = count_7 + 1
    if i == "2":
        count_8 = count_8 + 1
    if i == "3":
        count_9 = count_9 + 1
    if i == "4":
        count_10 = count_10 + 1
    if i == "5":
        count_11 = count_11 + 1
    if i == "6":
        count_12 = count_12 + 1

e01_2 = count_7 / N10
e02_2 = count_8 / N10
e03_2 = count_9 / N10
e04_2 = count_10 / N10
e05_2 = count_11 / N10
e06_2 = count_12 / N10

print("Start probability for loaded:  " + str(3 / 4) + '\n' + "End probability for loaded: " + str(2 / game_loaded_len))
print("Transition probability for loaded to fair: " + str(A10 / N10))
print("Emission Probability for loaded to fair: " + str(e01_2) + ", " + str(e02_2) + ", " + str(e03_2) + ", " +
      str(e04_2) + ", " + str(e05_2) + ", " + str(e06_2) + '\n')
print("Transition probability from loaded to loaded " + str(1 - A10 / N10 - 2 / game_loaded_len))
