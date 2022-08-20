#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from copy import deepcopy
from collections import deque
import heapq

#libraries for dialog window
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select maze image", filetypes=(("png files", "*.png"), ))
if len(file_path) == 0:
    exit("We hope you will manage to select a proper .png file next time.")

#read the image file
im = plt.imread(file_path)
#convert the image to greyscale
greyscale = np.mean(im[:,:,:3], axis = 2)
original_image = im[:,:,:3]

def create_maze(iba_steny:list, moves:list) -> list:
    '''Creates a maze with correct format from array of walls.'''
    '''Maze consists of cells with walls denoted by '#' and cells with integer values. Cost of traveling across a square
    is determined by it's integer value. The greater the value, the bigger path cost, effectively making the path stay
    exactly in the middle between walls'''
    # smoothing factor = how bad is it to travel closer to the wall
    smoothing_factor = 2
    res = deepcopy(iba_steny)
    akt = deque()
    steny = [[x, y, 2**30, smoothing_factor] for x in range(len(iba_steny)) for y in range(len(iba_steny[x])) if iba_steny[x][y] == '#']
    for stena in steny:
        akt.append(stena)
    # Find how far is an empty cell from the nearest wall with BFS
    while len(akt) > 0:
        tmp = akt.popleft()
        #print(type(tmp), tmp)
        for move in moves:
            if 0 <= tmp[0]+move[0] and tmp[0]+move[0] < len(res) and 0 <= tmp[1]+move[1] and tmp[1]+move[1] < len(res[0]):
                if res[tmp[0]+move[0]][tmp[1]+move[1]] == '.':
                    res[tmp[0]+move[0]][tmp[1]+move[1]] = tmp[2]//tmp[3]
                    akt.append([tmp[0]+move[0], tmp[1]+move[1], tmp[2]//tmp[3], tmp[3]])
    return res

# Filter walls from the greyscaled image
iba_steny = [['#' if x < 0.2 else '.' for x in riadok] for riadok in greyscale]
moves = [[-1,0],[1,0],[0,-1],[0,1]]
#create maze with the walls
bludisko = create_maze(iba_steny, moves)

def is_number(a):
    '''Check whether a is a proper number'''
    try:
        float(repr(a))
        return True
    except:
        return False

def vypis_tabulku(vypis):
    '''Black Magic, Nedotykat sa!!!'''
    # Fancy vypisanie Fancy outputu
    # Funkcia nie je crucial pre beh programu, ale vypisuje pekne bludiska
    s = [[str(e) for e in row] for row in [vypis[i] for i in range(len(vypis))]]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '  '.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

# zrata minimum z cisel v bludisku
minimum = min([x for riadok in bludisko for x in riadok if is_number(x)])
bludisko = [[x-minimum if is_number(x) else x for x in riadok] for riadok in bludisko]

def k_nearest_tiles(k:int, tile:list, maze:list, moves:list) -> list:
    '''Compute list of tiles with distance <= k from tile "tile" with manhatanian distance formula implemented via BFS'''
    res = []
    akt = deque()
    akt.append(tile+[0])
    while len(akt) > 0:
        tmp = akt.popleft()
        if tmp in res or tmp[2] > k:
            continue
        res.append(tmp)
        for move in moves:
            if 0 <= tmp[0]+move[0] and tmp[0]+move[0] < len(maze) and 0 <= tmp[1]+move[1] and tmp[1]+move[1] < len(maze[0]):
                if maze[tmp[0]+move[0]][tmp[1]+move[1]] != '#':
                    akt.append([tmp[0]+move[0], tmp[1]+move[1], tmp[2]+1])
    return res

def Najdi_cestu(maze:list, moves:list, start:list, finish:list):
    # Find the shortest path with Dijkstra's algorithm
    paths = [[[-1, -1] for _ in bludisko[0]] for _ in bludisko]
    akt = [[0]+start+start]
    while len(akt) > 0:
        tmp = heapq.heappop(akt)
        if paths[tmp[1]][tmp[2]] != [-1, -1]:
            continue
        paths[tmp[1]][tmp[2]] = [tmp[3], tmp[4]]
        if tmp[1:3] == finish:
            break
        for move in moves:
            if 0 <= tmp[1]+move[0] and tmp[1]+move[0] < len(maze) and 0 <= tmp[2]+move[1] and tmp[2]+move[1] < len(maze[0]):
                if maze[tmp[1]+move[0]][tmp[2]+move[1]] != '#':
                    heapq.heappush(akt, [tmp[0]+maze[tmp[1]][tmp[2]], tmp[1]+move[0], tmp[2]+move[1], tmp[1], tmp[2]])
    maze = deepcopy(maze)
    # Highlight path from start to finish with letter C for cesta
    while paths[finish[0]][finish[1]] != finish:
        maze[finish[0]][finish[1]] = 'C'
        finish = paths[finish[0]][finish[1]]
    maze[finish[0]][finish[1]] = 'C'
    # Make the path thiccest
    path_to_victory = [[x, y] for x in range(len(maze)) for y in range(len(maze[x])) if maze[x][y] == 'C']
    for tile in path_to_victory:
        zafarbi = k_nearest_tiles(1, tile, maze, moves)
        for tile_na_zafabenie in zafarbi:
            maze[tile_na_zafabenie[0]][tile_na_zafabenie[1]] = 'C'
    return maze

def close_enough(a, b):
    #Compute distance of two arrays
    error_rate = sum([abs(a[i]-b[i]) for i in range(len(a))])
    if error_rate < 0.1:
        return True
    return False

def Najdi_start(obrazok) -> list:
    '''Find start denoted by red pixel, there is som leeway if the image quaity is bad or you have budget image editor'''
    res = []
    for a in range(len(obrazok)):
        for b in range(len(obrazok[0])):
            if close_enough(list(obrazok[a][b][:3]), [1,0,0]):
                res = [a,b]
    return res

def Najdi_finish(obrazok) -> list:
    '''The same, but the finish is denoted by green pixel'''
    res = []
    for a in range(len(obrazok)):
        for b in range(len(obrazok[0])):
            if close_enough(list(obrazok[a][b][:3]), [0,1,0]):
                res = [a,b]
    return res

bludisko = Najdi_cestu(bludisko, moves, Najdi_start(im), Najdi_finish(im))

# Highlight the path
for i in range(len(original_image)):
    for j in range(len(original_image[i])):
        if bludisko[i][j] == 'C':
            original_image[i][j] = np.array([1.0,0.0,0.0])
        if bludisko[i][j] == '#':
            original_image[i][j] = np.array([0.0,0.0,0.0])

matplotlib.image.imsave('vyriesene.png', original_image)