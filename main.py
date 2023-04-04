import cv2
import pygame
import matplotlib.pyplot as plt
import numpy as np
import random
import math

def draw_text (text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

# Define constants
WINDOW_WIDTH = 1132
WINDOW_HEIGHT = 564

# Define colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Initialize Pygame
pygame.init()
pygame.display.set_caption("Spot the Difference")
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()

# Load images
menu = cv2.imread("menu.png")
highscore = cv2.imread("highscore.png")
choose_image = cv2.imread("choose_image.png")
choose_level = cv2.imread("choose_level.png")
input_name = cv2.imread("input_name.png")


#Get high score from txt
with open('highscore.txt', 'r') as file:
    lines = file.readlines()

highscore_array = []
for line in lines:
    highscore_array.append(line.strip())


#Generate different coord
random_x1 = random.randint(0, 566//2)
random_y1 = random.randint(0, 564//2)

random_x2 = random.randint(566//2, 566//2 + 566//4)
random_y2 = random.randint(0, 566//2)

random_x3 = random.randint(0, 283//2)
random_y3 = random.randint(283, 283 + 564//4)

random_x4 = random.randint(284, 284 + 566//4)
random_y4 = random.randint(283, 283 + 564//4)


#Draw random rectangles
w1 = random.randint(50, 70)
h1 = random.randint(50, 70)

#Converto to RGB
menu_rgb = cv2.cvtColor(menu, cv2.COLOR_BGR2RGB)
highscore_rgb = cv2.cvtColor(highscore, cv2.COLOR_BGR2RGB)
choose_image_rgb = cv2.cvtColor(choose_image, cv2.COLOR_BGR2RGB)
choose_level_rgb = cv2.cvtColor(choose_level, cv2.COLOR_BGR2RGB)
input_name_rgb = cv2.cvtColor(input_name, cv2.COLOR_BGR2RGB)


# Create list of difference points
diff_points = []
found_points = []

# Initialize score
score = 0

# Game loop
running = True
menu = True
highscore_flag = False
choose_image_flag = True
choose_level_flag = True
input_name_text = ''
input_name_flag = True
level = 1
rewrite = True

menu_font = pygame.font.SysFont("arialblack", 40)
while running:
    #Menu
    while (menu):
        screen.fill(BLACK)
        screen.blit(pygame.surfarray.make_surface(menu_rgb), (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                menu = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                #Start game
                if 415 <= pos[1] <= 498 and \
                    180 <= pos[0] <= 456:
                    menu = False

                #Show highscore
                if 415 <= pos[1] <= 498 and \
                    692 <= pos[0] <= 975:
                    menu = False
                    highscore_flag = True

        pygame.display.flip()
        clock.tick(24)

    #Display highscore
    while (highscore_flag):
        screen.blit(pygame.surfarray.make_surface(highscore_rgb), (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                highscore_flag = False
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if 42 <= pos[1] <= 132 and 32 <= pos[0] <= 114:
                    highscore_flag = False
                    menu = True
        y = 254
        rank = 1
        for name in highscore_array:
            font = pygame.font.Font(None, 72)
            text = font.render(str(rank) + ". " + name, True, (153, 76, 0))
            screen.blit(text, (321, y))
            y += 50
            rank += 1

        pygame.display.flip()
        clock.tick(24)

    if running == False:
        break

    #Choose image
    while (choose_image_flag):
        screen.fill(BLACK)
        screen.blit(pygame.surfarray.make_surface(choose_image_rgb), (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                choose_image_flag = False
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                choose_image_flag = False
                if 250 <= pos[1] <= 430 and 241 <= pos[0] <= 409:
                    img1 = cv2.imread("image1.jpg")
                elif 252 <= pos[1] <= 425 and 508 <= pos[0] <= 685:
                    img1 = cv2.imread("image_2.jpg")
                elif 249 <= pos[1] <= 432 and 770 <= pos[0] <= 949:
                    img1 = cv2.imread("image_3.jpg")
                else:
                    choose_image_flag = True

                if not choose_image_flag:
                    img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        pygame.display.flip()
        clock.tick(24)

    if running == False:
        break

    #Choose level
    while (choose_level_flag):
        screen.fill(BLACK)
        screen.blit(pygame.surfarray.make_surface(choose_level_rgb), (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                choose_level_flag = False
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                choose_level_flag = False
                if 274 <= pos[1] <= 401 and 139 <= pos[0] <= 252:
                    level = 1
                elif 275 <= pos[1] <= 393 and 356 <= pos[0] <= 462:
                    level = 2
                elif 282 <= pos[1] <= 400 and 556 <= pos[0] <= 671:
                    level = 3
                elif 277 <= pos[1] <= 398 and 756 <= pos[0] <= 874:
                    level = 4
                elif 270 <= pos[1] <= 400 and 941 <= pos[0] <= 1046:
                    level = 5
                else:
                    choose_level_flag = True
        pygame.display.flip()
        clock.tick(24)

    if running == False:
        break

    #Input name
    font = pygame.font.Font(None, 72)
    input_name_text_str = ""

    while (input_name_flag):
        screen.blit(pygame.surfarray.make_surface(input_name_rgb), (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                input_name_flag = False
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    input_name_flag = False
                elif event.key == pygame.K_BACKSPACE:
                    input_name_text_str = input_name_text_str[:-1]
                else:
                    input_name_text_str += event.unicode

        input_name_text = font.render(input_name_text_str, True, (153, 76, 0))
        screen.blit(input_name_text, (180, 280))

        pygame.display.flip()
        clock.tick(24)

    if running == False:
        break

    #Get Image
    img2 = img1.copy()

    #LEVEL 1
    if (level == 1):
        img2 = cv2.rectangle(img2, (random_x1, random_y1),
                             (random_x1 + w1, random_y1 + h1), (random_x1, 0, random_y1), -1)
        img2 = cv2.rectangle(img2, (random_x2, random_y2),
                             (random_x2 + w1 +25, random_y2 + h1 - 10), (random_x2, 0, random_y2), -1)
        img2 = cv2.rectangle(img2, (random_x3, random_y3),
                             (random_x3 + w1 + 10, random_y3 + h1 - 20), (random_x3, 0, random_y3), -1)
        img2 = cv2.rectangle(img2, (random_x4, random_y4),
                             (random_x4 + w1 - 10, random_y4 + h1 + 50), (random_x4, 0, random_y4), -1)

    #LEVEL 2
    if (level == 2):
        region_of_interest = img2[random_x1: random_x1+w1, random_y1: random_y1+h1]
        # flip vùng đã chọn theo trục dọc
        flipped_roi = cv2.flip(region_of_interest, 1)
        # gán lại vùng đã flip vào ảnh gốc
        img2[random_x1: random_x1+w1, random_y1: random_y1+h1] = flipped_roi

        region_of_interest = img2[random_x2: random_x2 + w1, random_y2: random_y2 + h1]
        # flip vùng đã chọn theo trục dọc
        flipped_roi = cv2.flip(region_of_interest, 1)
        # gán lại vùng đã flip vào ảnh gốc
        img2[random_x2: random_x2 + w1, random_y2: random_y2 + h1] = flipped_roi

        region_of_interest = img2[random_x3: random_x3 + w1, random_y3: random_y3 + h1]
        # flip vùng đã chọn theo trục dọc
        flipped_roi = cv2.flip(region_of_interest, 1)
        # gán lại vùng đã flip vào ảnh gốc
        img2[random_x3: random_x3 + w1, random_y3: random_y3 + h1] = flipped_roi

        region_of_interest = img2[random_x4: random_x4 + w1, random_y4: random_y4 + h1]
        # flip vùng đã chọn theo trục dọc
        flipped_roi = cv2.flip(region_of_interest, 1)
        # gán lại vùng đã flip vào ảnh gốc
        img2[random_x4: random_x4 + w1, random_y4: random_y4 + h1] = flipped_roi

    #LEVEL 3
    if (level == 3):
        img2[random_y1:random_y1 + h1, random_x1:random_x1 + w1] =\
            cv2.GaussianBlur(img2[random_y1:random_y1 + h1, random_x1:random_x1 + w1], (15, 15), 0)
        img2[random_y2:random_y2 + h1, random_x2:random_x2 + w1] =\
            cv2.GaussianBlur(img2[random_y2:random_y2 + h1, random_x2:random_x2 + w1], (15, 15), 0)
        img2[random_y3:random_y3 + h1, random_x3:random_x3 + w1] = \
            cv2.GaussianBlur(img2[random_y3:random_y3 + h1, random_x3:random_x3 + w1], (15, 15), 0)
        img2[random_y4:random_y4 + h1, random_x4:random_x4 + w1] = \
            cv2.GaussianBlur(img2[random_y4:random_y4 + h1, random_x4:random_x4 + w1], (15, 15), 0)

    #LEVEL 4
    if (level == 4):
        template = cv2.imread("template1.jpg")
        template2 = cv2.imread("template2.jpg")
        result = cv2.matchTemplate(img1, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.42
        loc = np.where(result >= threshold)
        mask = np.zeros_like(img1)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(mask, pt, (pt[0] + 39, pt[1] + 28), (255, 255, 255), -1)
        img2 = cv2.inpaint(img1, mask[..., 0], 3, cv2.INPAINT_TELEA)

        result = cv2.matchTemplate(img1, template2, cv2.TM_CCOEFF_NORMED)
        threshold = 0.48
        loc = np.where(result >= threshold)
        mask3 = np.zeros_like(img1)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(mask3, pt, (pt[0] + 56, pt[1] + 39), (255, 255, 255), -1)
        img2 = cv2.inpaint(img2, mask3[..., 0], 3, cv2.INPAINT_TELEA)


    #LEVEL 5
    if level == 5:
        # Chuyển ảnh sang ảnh xám
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # Phát hiện cạnh bằng canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Tìm contour của các vật thể
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Chọn các contour có diện tích lớn hơn ngưỡng cố định và lưu vào list
        contours_selected= []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 600:
                contours_selected.append(cnt)
        # Sắp xếp các contour theo diện tích giảm dần
        contours_selected.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        # Tạo mask và tô màu vào các vùng contour được chọn
        mask = np.zeros(img1.shape[:2], dtype=np.uint8)
        for cnt in contours_selected[:5]:
            cv2.drawContours(mask, [cnt], 0, 255, -1)
        # Tô màu vào các vùng mask
        img2 = img1.copy()
        img2[mask == 255] = [0, 200, 0]  # Tô màu đỏ

    # Convert images to RGB format for display in Pygame
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Convert images to gray
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Get differences between images
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Find contours of differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 30:
            center = (x + int(w / 2), y + int(h / 2))
            center = (center[1], center[0])
            bbox = (y, x, w, h)
            diff_points.append({"center": center, "bbox": bbox})

    if (rewrite):
        cv2.imwrite('diff_image.jpg', cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE))
        rewrite = False

    # Draw images and score
    screen.fill(BLACK)
    screen.blit(pygame.surfarray.make_surface(img1_rgb), (0, 0))
    screen.blit(pygame.surfarray.make_surface(img2_rgb), (WINDOW_WIDTH // 2, 0))
    # Draw circle around difference point
    for point in found_points:
        pygame.draw.circle(screen, (255, 0, 0), point["center"], 10)

    font = pygame.font.Font(None, 36)
    text = font.render("Score: " + str(score), True, (255, 255, 255))
    screen.blit(text, (10, WINDOW_HEIGHT - 40))

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if mouse click is within difference point bounding box
            pos = pygame.mouse.get_pos()
            for point in diff_points:
                if point["bbox"][0] <= pos[0] < point["bbox"][0] + point["bbox"][2] \
                        and point["bbox"][1] <= pos[1] < point["bbox"][1] + point["bbox"][3]:
                    found_points.append(point)
                    # Increment score and remove point from list
                    score += 1

                    diff_points.remove(point)
                    break

    # Draw circles around difference points
    for point in diff_points:
        is_intersect = False
        for other_point in diff_points:
            if point != other_point:
                dist = math.sqrt((point["center"][0] - other_point["center"][0]) ** 2 + (
                            point["center"][1] - other_point["center"][1]) ** 2)
                if dist < 50:
                    is_intersect = True
                    break
        if not is_intersect:
            pygame.draw.circle(screen, (255, 0, 0), point["center"], 50, 5)

    # Update display
    pygame.display.flip()

    # Limit frame rate to 60 FPS
    clock.tick(60)

# Exit game
pygame.quit()
cv2.destroyAllWindows()

