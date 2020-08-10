import time
import win32gui
import random
import os
import datetime as dt
import itertools

import numpy as np
from PIL import ImageGrab
import cv2
import pyautogui


class Square:

    def __init__(self, value, position, empty_count, flag_count, empty_pos=None, flag_pos=None):
        self.value = value
        self.pos = position
        self.emptys = empty_count
        self.flags = flag_count
        self.empty_pos = empty_pos
        self.flag_pos = flag_pos
        self.remaining = value - flag_count
        if self.emptys > 0:
            self.mine_prob = self.remaining / self.emptys
        else:
            self.mine_prob = 0

    def __repr__(self):
        return '{} at {} with {} empty, {} flags'.format(self.value, self.pos, self.emptys, self.flags)

    def __str__(self):
        return '{} at {} with {} empty, {} flags'.format(self.value, self.pos, self.emptys, self.flags)

    def common_emptys(self, other):
        if len(self.empty_pos) >= len(other.empty_pos):
            return [pos for pos in other.empty_pos if pos in self.empty_pos]
        else:
            return [pos for pos in self.empty_pos if pos in other.empty_pos]

    def common_flags(self, other):
        if len(self.flag_pos) >= len(other.flag_pos):
            return [pos for pos in other.flag_pos if pos in self.flag_pos]
        else:
            return [pos for pos in self.flag_pos if pos in other.flag_pos]

    def diff_emptys(self, other):
        return [pos for pos in self.empty_pos if pos not in other.empty_pos]

    def diff_flags(self, other):
        return [pos for pos in self.flag_pos if pos not in other.flag_pos]

def callback(hwnd, name):
    global WINDOWCOORD
    if win32gui.GetWindowText(hwnd) == name:
        rect = win32gui.GetWindowRect(hwnd)
        WINDOWCOORD = (rect[0], rect[1], rect[2], rect[3])

def get_window_coords(name):
    global WINDOWCOORD
    WINDOWCOORD = None
    win32gui.EnumWindows(callback, name)
    return WINDOWCOORD

def draw_lines(img, lines, colour=[0, 255, 0], linewidth=2):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), colour, linewidth)
    except:
        pass

def process_img(original_img, thresh1=200, thresh2=300):
    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, thresh1, thresh2)
    #processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    return processed_img

def coord_to_index(coord, min_coord, max_coord, gridsize=16, size=None):
    if size == None:
        size = (int(round((max_coord[1] - min_coord[1]) / 16) + 1), int(round((max_coord[0] - min_coord[0]) / 16) + 1))
    else:
        size = size
    # coords are (x, y), size is rows x columns
    x_i = int(round((coord[0] - min_coord[0]) / gridsize))
    y_i = int(round((coord[1] - min_coord[1]) / gridsize))
    return x_i, y_i

def match_templates(img, gray_img):
    # hiscore
    template = cv2.imread('templates/hiscore.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    hiscore_loc = np.where(res >= threshold)
    for pt in zip(*hiscore_loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255, 255, 0), 1)
    if len(hiscore_loc[0]) > 0:
        return None, None, None, 'hiscore', None, None
    # leaderboard
    template = cv2.imread('templates/leaderboard.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    leaderboard_loc = np.where(res >= threshold)
    for pt in zip(*leaderboard_loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255, 255, 0), 1)
    if len(leaderboard_loc[0]) > 0:
        return None, None, None, 'leaderboard', None, None
    # smiley face
    template = cv2.imread('templates/smiley_face.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.92
    smiley_face_loc = np.where(res >= threshold)
    for pt in zip(*smiley_face_loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255, 255, 0), 1)
    # dead face
    template = cv2.imread('templates/dead_face.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.92
    dead_face_loc = np.where(res >= threshold)
    for pt in zip(*dead_face_loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255, 255, 0), 1)
    # win face
    template = cv2.imread('templates/win_face.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.92
    win_face_loc = np.where(res >= threshold)
    for pt in zip(*win_face_loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255, 255, 0), 1)
    # mines
    mine_locs = []
    template = cv2.imread('templates/mine.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.92
    mine_loc = np.where(res >= threshold)
    for pt in zip(*mine_loc[::-1]):
        mine_locs.append(pt)
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 1)
    # flags
    flag_locs = []
    template = cv2.imread('templates/flag.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.92
    flag_loc = np.where(res >= threshold)
    for pt in zip(*flag_loc[::-1]):
        flag_locs.append(pt)
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 1)
    # question
    question_locs = []
    template = cv2.imread('templates/question.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    question_loc = np.where(res >= threshold)
    for pt in zip(*question_loc[::-1]):
        question_locs.append(pt)
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 128, 255), 1)
    # empty
    empty_locs = []
    template = cv2.imread('templates/empty.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    empty_loc = np.where(res >= threshold)
    for pt in zip(*empty_loc[::-1]):
        empty_locs.append(pt)
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (128, 0, 255), 1)
    # pressed
    pressed_locs = []
    template = cv2.imread('templates/pressed.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    pressed_loc = np.where(res >= threshold)
    for pt in zip(*pressed_loc[::-1]):
        pressed_locs.append(pt)
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 255, 0), 1)
    # counters
    count_locs = [[], [], [], [], [], [], [], [], [], []]
    for count in range(10):
        template = cv2.imread('templates/count{}.png'.format(count), 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.98
        count_loc = np.where(res >= threshold)
        for pt in zip(*count_loc[::-1]):
            count_locs[count].append(pt)
            cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255, 255, 0), 1)
    # numbers
    number_locs = [[], [], [], [], [], [], [], []]
    for num in range(1, 9):
        template = cv2.imread('templates/num{}.png'.format(num), 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
        num_loc = np.where(res >= threshold)
        for pt in zip(*num_loc[::-1]):
            number_locs[num-1].append(pt)
            cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 1)
    
    all_coords = mine_locs + flag_locs + question_locs + empty_locs + pressed_locs + [x for sub in number_locs for x in sub]
    if all_coords == []:
        cv2.putText(img, 'CAN\'T SEE FULL SCREEN', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return None, None, None, 'blocked', None, None
    min_coord = min(all_coords, key=lambda t: t[0]+t[1])
    grid_coord = min_coord
    max_coord = max(all_coords, key=lambda t: t[0]+t[1])
    minx, miny, maxx, maxy = min_coord[0], min_coord[1], max_coord[0], max_coord[1]
    cv2.rectangle(img, (minx - 1, miny - 1), (maxx + 17, maxy + 17), (255, 0, 0), 1)
    grid_shape = (int(round((maxy - miny) / 16) + 1), int(round((maxx - minx) / 16) + 1))
    grid = np.zeros((grid_shape), dtype=np.int64)
    # get counter values
    try:
        # tuple with value, x location
        count_digits = []
        for i in range(len(count_locs)):
            for loc in count_locs[i]:
                count_digits.append((i, loc[0]))
        count_digits.sort(key=lambda t: t[1])
        mines_left = 100 * (count_digits[0][0]) + 10 * (count_digits[1][0]) + (count_digits[2][0])
        timer = 100 * (count_digits[3][0]) + 10 * (count_digits[4][0]) + (count_digits[5][0])
        cv2.putText(img, '{}'.format(mines_left), (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img, '{}'.format(timer), (gray_img.shape[1] - 52, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        got_counter = True
    except:
        cv2.putText(img, 'Can\'t detect counter', (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        got_counter = False
    # read state
    if len(smiley_face_loc[0]) > 0:
        game_state = 'playing'
        face_coord = (smiley_face_loc[1][0], smiley_face_loc[0][0])
        cv2.putText(img, 'smiley', face_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    elif len(dead_face_loc[0]) > 0:
        game_state = 'lose'
        face_coord = (dead_face_loc[1][0], dead_face_loc[0][0])
        cv2.putText(img, 'lose', face_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    elif len(win_face_loc[0]) > 0:
        game_state = 'win'
        face_coord = (win_face_loc[1][0], win_face_loc[0][0])
        cv2.putText(img, 'win', face_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    else:
        game_state = 'waiting'
        face_coord = None
    # plot grid
    try:
        for loc in flag_locs:
            index = coord_to_index(loc, min_coord, max_coord, 16, size=grid_shape)
            grid[index[1], index[0]] = -2
        for loc in question_locs:
            index = coord_to_index(loc, min_coord, max_coord, 16, size=grid_shape)
            grid[index[1], index[0]] = -3
        for loc in pressed_locs:
            index = coord_to_index(loc, min_coord, max_coord, 16, size=grid_shape)
            grid[index[1], index[0]] = -1
        for i in range(len(number_locs)):
            for loc in number_locs[i]:
                index = coord_to_index(loc, min_coord, max_coord, 16, size=grid_shape)
                grid[index[1], index[0]] = i + 1
    except:
        cv2.putText(img, 'CAN\'T SEE FULL SCREEN', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return None, None, None, 'blocked', None, None
    if got_counter:
        return grid, mines_left, timer, game_state, grid_coord, face_coord
    else:
        return grid, None, None, game_state, grid_coord, face_coord

def rel_coord(coord, ref_coord):
    return (ref_coord[0] + coord[0], ref_coord[1] + coord[1])

def click_grid(index, grid_coord, button='left', instant=True):
    # index in (x, y)
    x, y = rel_coord((index[0]*16 + 8, index[1]*16 + 8), grid_coord)
    if instant:
        pyautogui.click(x=x, y=y, button=button)
    else:
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.click(button=button)

def middle_click_grid(index, grid_coord, instant=True):
    x, y = rel_coord((index[0]*16 + 8, index[1]*16 + 8), grid_coord)
    if instant:
        pyautogui.mouseDown(x=x, y=y, button='left')
        pyautogui.mouseDown(button='right')
        pyautogui.mouseUp(button='left')
        pyautogui.mouseUp(button='right')
    else:
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.mouseDown(button='left')
        pyautogui.mouseDown(button='right')
        pyautogui.mouseUp(button='left')
        pyautogui.mouseUp(button='right')

def neighbouring(index, grid):
    # index is (x, y), grid is (y, x)
    values = [None] * 8
    grid_size = grid.shape
    if index[1] > 0:
        if index[0] > 0:
            values[0] = grid[index[1] - 1, index[0] - 1]
        values[1] = grid[index[1] - 1, index[0]]
        if index[0] < grid_size[1] - 1:
            values[2] = grid[index[1] - 1, index[0] + 1]
    if index[0] > 0:
        values[3] = grid[index[1], index[0] - 1]
    if index[0] < grid_size[1] - 1:
        values[4] = grid[index[1], index[0] + 1]
    if index[1] < grid_size[0] - 1:
        if index[0] > 0:
            values[5] = grid[index[1] + 1, index[0] - 1]
        values[6] = grid[index[1] + 1, index[0]]
        if index[0] < grid_size[1] - 1:
            values[7] = grid[index[1] + 1, index[0] + 1]
    return values

def neighbour_index_to_index(center_i, neighbour_i):
    indices = []
    for num in neighbour_i:
        if num == 0:
            indices.append((center_i[0] - 1, center_i[1] - 1))
        elif num == 1:
            indices.append((center_i[0], center_i[1] - 1))
        elif num == 2:
            indices.append((center_i[0] + 1, center_i[1] - 1))
        elif num == 3:
            indices.append((center_i[0] - 1, center_i[1]))
        elif num == 4:
            indices.append((center_i[0] + 1, center_i[1]))
        elif num == 5:
            indices.append((center_i[0] - 1, center_i[1] + 1))
        elif num == 6:
            indices.append((center_i[0], center_i[1] + 1))
        elif num == 7:
            indices.append((center_i[0] + 1, center_i[1] + 1))
    return indices

def allUnique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)

def non_overlap_combinations(common_squares_list):
    valid_combinations = []
    for l in range(2, len(common_squares_list) + 1):
        for comb in itertools.combinations(common_squares_list, l):
            flattened = [pos for sub in comb for pos in sub]
            if allUnique(flattened):
                valid_combinations.append(comb)
    return valid_combinations

def auto_play(grid, m_left, timer, state, window_coord, rel_grid_coord, rel_face_coord, sim=False):
    global TESTSTEP
    debug_timer = time.time()
    if state == 'blocked':
        return
    if state == 'hiscore':
        pyautogui.typewrite('Mr Roboto', interval=0.1)
        pyautogui.typewrite(['enter'])
        return
    if state == 'leaderboard':
        pyautogui.typewrite(['enter'])
        return
    if m_left == None or timer == None or rel_grid_coord == None or rel_face_coord == None:
        return
    grid_coord = rel_coord(rel_grid_coord, window_coord)
    face_coord = rel_coord(rel_face_coord, window_coord)
    grid_size = grid.shape
    if state == 'playing':
        if not np.any(grid):
            print('START GAME')
            click_grid((0, 0), grid_coord)
            #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'first move'))
            return
        empty_locs = np.where(grid == 0)
        empty_locs = list(zip(*empty_locs[::-1]))
        question_locs = np.where(grid == -3)
        question_locs = list(zip(*question_locs[::-1]))
        # right click question marks
        if len(question_locs) > 0:
            for index in question_locs:
                click_grid(index, grid_coord, button='right')
            #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'clear questions'))
            return
        # click remaining squares if no mines left
        if m_left == 0:
            for index in empty_locs:
                click_grid(index, grid_coord, button='left')
            #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'click remaining squares'))
            return
        open_grids = []
        to_clear = []
        # for loop collect info, get all the obvious ones
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                cur_value = grid[row, col]
                cur_i = (col, row)
                if cur_value > 0:
                    nvalues = neighbouring(cur_i, grid)
                    empty_count = nvalues.count(0)
                    flag_count = nvalues.count(-2)
                    if empty_count > 0:
                        empty_i = [i for i, val in enumerate(nvalues) if val == 0]
                        empty_i = neighbour_index_to_index(cur_i, empty_i)
                        # place flags
                        if empty_count + flag_count == cur_value:
                            m_left -= empty_count
                            for i in empty_i:
                                grid[i[::-1]] = -2
                                click_grid(i, grid_coord, button='right')
                            #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'place flags'))
                            if not sim:
                                auto_play(grid, m_left, timer, state, window_coord, rel_grid_coord, rel_face_coord)
                            return
                        if flag_count == cur_value:
                            to_clear.append((empty_count, cur_i, empty_i))
                        #open_grids.append([cur_value, cur_i, empty_count, flag_count, empty_i])
                        open_grids.append(Square(cur_value, cur_i, empty_count, flag_count, empty_i))
        # clear efficiently
        if len(to_clear) > 0:
            single_clear_i = []
            max_clear_i = (0, (0, 0))
            for i in to_clear:
                if i[0] == 1:
                    single_clear_i.append(i)
                if i[0] > max_clear_i[0]:
                    max_clear_i = i
            if max_clear_i[0] > 1:
                middle_click_grid(max_clear_i[1], grid_coord)
                #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'both click'))
                return
            else:
                for clear_i in single_clear_i:
                    for i in clear_i[2]:
                        click_grid(i, grid_coord, button='left')
                #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'single clicks'))
                return
        # simple deduction
        third_deduct = {}   # dictionary of (remaining, position): squares that share same empty locations, common squares, current empty pos
        for i in range(len(open_grids)):
            third_deduct[(open_grids[i].remaining, open_grids[i].pos)] = []
            temp_nums = open_grids[:]
            temp_nums.pop(i)
            # for num in open_grids except itself and only if its location is within 2 x/y
            for num in [sq for sq in temp_nums if (abs(sq.pos[0] - open_grids[i].pos[0]) <= 2) and (abs(sq.pos[1] - open_grids[i].pos[1]) <= 2)]:
                common_squares = open_grids[i].common_emptys(num)
                if open_grids[i].empty_pos != num.empty_pos:
                    # if common squares are equal to current square
                    if num.empty_pos == common_squares and open_grids[i].emptys > num.emptys:
                        if num.empty_pos not in (sq.empty_pos for sq in (item[0] for item in third_deduct[(open_grids[i].remaining, open_grids[i].pos)])):
                            third_deduct[(open_grids[i].remaining, open_grids[i].pos)].append((num, common_squares, open_grids[i].empty_pos))
                        if open_grids[i].remaining == num.remaining:
                            for index in (t for t in open_grids[i].empty_pos if t not in num.empty_pos):
                                click_grid(index, grid_coord, button='left')
                            #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'simple deduction 1'))
                            return
                        if open_grids[i].emptys - num.emptys == open_grids[i].remaining - num.remaining:
                            for index in (t for t in open_grids[i].empty_pos if t not in num.empty_pos):
                                click_grid(index, grid_coord, button='right')
                            #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'simple deduction 2'))
                            return
                    # else if the emptys are different
                    if open_grids[i].empty_pos != common_squares:
                        min_mines_in_common = open_grids[i].remaining - (open_grids[i].emptys - len(common_squares))
                        if min_mines_in_common == num.remaining:
                            for index in (t for t in open_grids[i].empty_pos if t not in common_squares):
                                click_grid(index, grid_coord, button='right')
                            #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'simple deduction 3'))
                            return
        # 3rd level deduction
        for cur in third_deduct:
            if len(third_deduct[cur]) > 0:
                #print('number of sets of common squares:', len(third_deduct[cur]))
                sq_com_and_emptypos = list(zip(*third_deduct[cur]))
                empty_pos = sq_com_and_emptypos[2][0]
                common_sq_dict = {}
                for i in range(len(sq_com_and_emptypos[0])):
                    common_sq_dict[tuple(sq_com_and_emptypos[1][i])] = sq_com_and_emptypos[0][i]
                combinations = non_overlap_combinations(sq_com_and_emptypos[1])
                for comb in combinations:
                    neigh_remaining_sum = sum((common_sq_dict[tuple(common)].remaining for common in comb))
                    common_squares = [item for sub in comb for item in sub]
                    cur_remaining = cur[0]
                    if len(empty_pos) > len(common_squares):
                        if cur_remaining == neigh_remaining_sum:
                            #print(cur[1], cur_remaining, neigh_remaining_sum, comb)
                            for pos in (i for i in empty_pos if i not in common_squares):
                                click_grid(pos, grid_coord, button='left')
                            #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, '3rd level deduction 1'))
                            return
                        if cur_remaining - neigh_remaining_sum == len(empty_pos) - len(common_squares):
                            #print(cur[1], cur_remaining, neigh_remaining_sum, comb)
                            for pos in (i for i in empty_pos if i not in common_squares):
                                click_grid(pos, grid_coord, button='right')
                            #print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, '3rd level deduction 2'))
                            return
        # simulation / mine counting
        
        # probablity
        if TESTSTEP:
            TESTSTEP = False
            # first take chance - if common squares of one does not overlap with any other
            
        # start over/give up
##        print('give up')
##        with open('Record.txt', 'a') as record_file:
##            record_file.write('GIVE UP: Time = {:3d}, Mines left = {:3d}, {}\n'.format(timer, m_left,
##                                                                                     dt.datetime.today().strftime('%d-%m-%y %H:%M:%S')))
##        print('Thinking time: {:.8f} ({})'.format(time.time() - debug_timer, 'give up'))
##        pyautogui.typewrite(['f2'])
                    
##    elif state == 'lose':
##        pyautogui.moveTo(rel_coord((12, 12), face_coord), duration=0.5)
##        pyautogui.click(button='left')
##        with open('Record.txt', 'a') as record_file:
##            record_file.write('LOSE   : Time = {:3d}, Mines left = {:3d}, {}\n'.format(timer, m_left,
##                                                                                     dt.datetime.today().strftime('%d-%m-%y %H:%M:%S')))
##        return
    elif state == 'win':
        pyautogui.moveTo(rel_coord((12, 12), face_coord), duration=0.5)
        pyautogui.click(button='left')
        with open('Record.txt', 'a') as record_file:
            record_file.write('WIN    : Time = {:3d}, Mines left = {:3d}, {}\n'.format(timer, m_left,
                                                                                     dt.datetime.today().strftime('%d-%m-%y %H:%M:%S')))
        return

def execute():
    global fps, AUTO, TESTSTEP
    while True:
        start_time_fps = time.time()
        window_coords = get_window_coords('Minesweeper')
        if window_coords == None:
            os.startfile('Minesweeper.exe')
            time.sleep(0.2)
            window_coords = get_window_coords('Minesweeper')
            
        window_coords = (window_coords[0]+6, window_coords[1]+48,
                         window_coords[2]-3, window_coords[3]-3)
        screen = np.array(ImageGrab.grab(bbox=window_coords))
        full_image = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        grid_array, mines_left, timer, state, grid_coord, face_coord = match_templates(full_image, gray_image)
        #print(grid_array, mines_left, timer, state)

        # Focus "Analysis Window" before pressing keys
        if cv2.waitKey(1) & 0xFF == ord('a'):
            if AUTO:
                AUTO = False
            else:
                AUTO = True
        if cv2.waitKey(1) & 0xFF == ord('1'):
            if not AUTO:
                AUTO = True
        if cv2.waitKey(1) & 0xFF == ord('2'):
            if AUTO:
                AUTO = False
        if cv2.waitKey(1) & 0xFF == ord('t'):
            if not TESTSTEP:
                TESTSTEP = True
        cv2.putText(full_image, 'AUTO {}'.format(AUTO), (gray_image.shape[1] - 80, gray_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        if AUTO:
            #start = time.time()
            auto_play(grid_array, mines_left, timer, state, (window_coords[0], window_coords[1]), grid_coord, face_coord)
            #print(time.time() - start)

        cv2.putText(full_image, 'FPS: {:.2f}'.format(fps), (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.imshow('Analysis Window', full_image)
        fps = 1 / (time.time() - start_time_fps)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Quit')
            cv2.destroyAllWindows()
            break

def main():
    execute()

pyautogui.PAUSE = 0.001
fps = 0
AUTO = False
TESTSTEP = False
main()
