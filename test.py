import pygame
import cv2
import numpy as np
import sys

img = pygame.image.load('test.jpg')
pygame.display.set_mode(img.get_size())
pygame.display.set_caption('Image')

img_np = pygame.surfarray.array3d(img)
img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
template = cv2.imread('template.jpg', 0)


result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc

thresh = cv2.threshold(result, 0.8, 255, cv2.THRESH_BINARY)[1]
mask = np.zeros(img_gray.shape, np.uint8)
mask[top_left[1]:top_left[1]+template.shape[0], top_left[0]:top_left[0]+template.shape[1]] = thresh

dst = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)
dst_surface = pygame.surfarray.make_surface(dst)

pygame.display.get_surface().blit(dst_surface, (0, 0))
pygame.display.update()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
