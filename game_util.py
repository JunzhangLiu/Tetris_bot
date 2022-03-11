from cProfile import label
from PIL import Image
import win32gui, win32process, win32ui, win32con
from win32com.client import GetObject
import numpy as np
import ctypes
import ctypes.wintypes
import time
from subprocess import check_output


def get_screenshot_args():
    dc = win32gui.GetDC(0)
    dc_obj=win32ui.CreateDCFromHandle(dc)
    c_dc=dc_obj.CreateCompatibleDC()
    args = {}
    args['dc'] = dc
    args['dc_obj'] = dc_obj
    args['c_dc'] = c_dc
    return args

def screenshot(args,w=700,h=770,hchannels=3):
    dc,dc_obj,c_dc = args['dc'],args['dc_obj'],args['c_dc']
    bit_map = win32ui.CreateBitmap()
    bit_map.CreateCompatibleBitmap(dc_obj,w, h)
    c_dc.SelectObject(bit_map)
    c_dc.BitBlt((0,0),(w, h) , dc_obj, (620,110), win32con.SRCCOPY)    
    bit_map_str = bit_map.GetBitmapBits(True)
    bit_map_info = bit_map.GetInfo()
    img = Image.frombuffer('RGB',(bit_map_info['bmWidth'], bit_map_info['bmHeight']),bit_map_str, 'raw', 'BGRX', 0, 1)
    img = img.convert('L')
    win32gui.DeleteObject(bit_map.GetHandle())
    img = np.array(img,dtype=np.uint8)
    img[200:,:157] = 0
    img[630:,545:] = 0
    return img

def detect_blocks(img, trianing = False):
    cols = []
    # img = np.array(img)
    board=img[76:-4,168:512]

    grid_h=28
    grid_w=28
    boundary_w=7
    boundary_h=7

    y_l=0
    for i in range(20):
        cols.append([])
        x_l=0
        y_r=y_l+grid_h
        if i == 9:
            boundary_h=6
        if i == 12:
            boundary_h=7
        for j in range(10):
            x_r=x_l+grid_w
            grid = board[y_l:y_r,x_l:x_r]
            if grid[-1,-1]>=230:
                status = 2
            elif grid[-1,-1]>=10:
                status=1
            else:
                status=0
            cols[-1].append(status)
            # Image.fromarray(grid).save(str('./grids/{:d}-{:d}.png'.format(i,j)))
            # print(grid[-2,-2],i,j,end='\t')
            x_l=x_r+boundary_w
        y_l=y_r+boundary_h
    blocks = np.array(cols,np.uint8)
    return remove_floating(blocks)

def remove_floating(blocks):
    for i in range(1,blocks.size[0]):
        if 1 not in blocks[i-1]:
            blocks[i][blocks[i]==1]=0
    return blocks

def death(blocks):
    if 2 in blocks[0] and 1 in blocks[1]:
        return True
    return False

def label_blocks(img):
    # blocks = np.array(img,dtype=np.uint8)
    board=img[76:-4,168:512]

    grid_h=28
    grid_w=28
    boundary_w=7
    boundary_h=7

    y_l=0
    for i in range(20):
        x_l=0
        y_r=y_l+grid_h
        if i == 9:
            boundary_h=6
        if i == 12:
            boundary_h=7
        for j in range(10):
            x_r=x_l+grid_w
            grid = board[y_l:y_r,x_l:x_r]
            if grid[-1,-1]>=230:
                grid[:grid_w//2,:grid_h//2]=250
            elif grid[-1,-1]>=10:
                grid[:grid_w//2,:grid_h//2]=70
            x_l=x_r+boundary_w
        y_l=y_r+boundary_h
    return board

# time.sleep(1)
# win_handle = 0
# args = get_screenshot_args()

# i=0
# img = screenshot(args)
# grid = detect_blocks(img)
# while not death(grid):
#     img1 = screenshot(args)
#     grid1 = detect_blocks(img1)
#     while (grid1==grid).all():
#         img1 = screenshot(args)
#         grid1 = detect_blocks(img1)
#     grid = grid1
#     Image.fromarray(label_blocks(img1)).save('imgs/'+str(i)+'.png')
#     i+=1
# for i in range(100):
#     time.sleep(1)
#     img = screenshot(args)
#     Image.fromarray(labeled_blocks).save('./imgs/'+str(i)+'.png')
