import pygame
import random
from typing import List, Tuple


# Initialize pygame
pygame.init()

# Set the window size
window_size = (400, 600)

# Create the window
screen = pygame.display.set_mode(window_size)

# Set the title
pygame.display.set_caption("Pac-Man")

# Set the background color
bg_color = (0, 0, 0)

# Load the images
pacman_image = pygame.image.load("pacman.png")
pacman_image = pygame.transform.scale(pacman_image, (32, 32))
ghost_image = pygame.image.load("ghost.png")
ghost_image = pygame.transform.scale(ghost_image, (32, 32))

# chage the color of the ghost
ghost_1 = ghost_image.copy()
ghost_1.fill((255, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
ghost_2 = ghost_image.copy()
ghost_2.fill((0, 255, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
ghost_3 = ghost_image.copy()
ghost_3.fill((0, 0, 255, 255), special_flags=pygame.BLEND_RGBA_MULT)
ghost_4 = ghost_image.copy()
ghost_4.fill((255, 255, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)

# Create four ghost images to select from, tinted different colors
ghost_images = [ghost_1, ghost_2, ghost_3, ghost_4]

# Set the movement speed of Pac-Man
pacman_speed = 5

# Setup a bunch of ghosts
class Sprite:
    def __init__(self, loc: Tuple[int, int], can_bounce: bool=True):
        self.loc = loc
        self.direction = (random.random() * 2 - 1, random.random() * 2 - 1)
        self.can_bounce = can_bounce

    def move(self):
        self.loc = (self.loc[0] + self.direction[0], self.loc[1] + self.direction[1])

        if self.can_bounce:
            if self.loc[0] < 0 or self.loc[0] > window_size[0] - 32:
                self.direction = (-self.direction[0], self.direction[1])
            if self.loc[1] < 0 or self.loc[1] > window_size[1] - 32:
                self.direction = (self.direction[0], -self.direction[1])
        else:
            if self.loc[0] < 0:
                self.loc = (window_size[0] - 32, self.loc[1])
            if self.loc[0] > window_size[0] - 32:
                self.loc = (0, self.loc[1])
            if self.loc[1] < 0:
                self.loc = (self.loc[0], window_size[1] - 32)
            if self.loc[1] > window_size[1] - 32:
                self.loc = (self.loc[0], 0)

    def collides(self, pacman_loc: Tuple[int, int]) -> bool:
        return self.loc[0] <= pacman_loc[0] <= self.loc[0] + 32 and self.loc[1] <= pacman_loc[1] <= self.loc[1] + 32

ghosts: List[Sprite] = [Sprite((100, 100)), Sprite((200, 200)), Sprite((300, 300))]

pacman = Sprite((200, 300), can_bounce=False)
# Pacman starts still, unlike the ghosts
pacman.direction = (0, 0)

# GAME SCORE
score = 0

# keep track of the number of times the ghost has been caught
caught = 0

# display the score
font = pygame.font.SysFont("Arial", 24)

# Run the game loop
running = True
frames = 0

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            # Handle arrow key presses
            elif event.key == pygame.K_LEFT:
                pacman.direction = (-pacman_speed, 0)
            elif event.key == pygame.K_RIGHT:
                pacman.direction = (pacman_speed, 0)
            elif event.key == pygame.K_UP:
                pacman.direction = (0, -pacman_speed)
            elif event.key == pygame.K_DOWN:
                pacman.direction = (0, pacman_speed)

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                pacman.direction = (0, 0)
    # Update game state
    # Move the ghost
    for ghost in ghosts:
        ghost.move()
    # and pacman
    pacman.move()

    # Every 5 seconds, add a new ghost
    if frames % 300 == 0:
        ghosts.append(Sprite((random.random() * window_size[0], random.random() * window_size[1])))

    # Check for collisions
    if any(ghost.collides((pacman.loc[0], pacman.loc[1])) for ghost in ghosts):
        # Add to the score
        score += 1
        # Add to the number of times the ghost has been caught
        caught += 1

        # display the score
        score_text = font.render("Score: " + str(score), True, (255, 255, 255))

        # Pac-Man and the ghost are colliding, end the game
        running = False

    # Draw the screen
    screen.fill(bg_color)
    # Draw Pac-Man at the new position
    screen.blit(pacman_image, (pacman.loc[0] - 32, pacman.loc[1] - 32))
    # Draw the ghosts at the new positions
    for i, ghost in enumerate(ghosts):
        screen.blit(ghost_images[i % len(ghost_images)], (ghost.loc[0], ghost.loc[1]))

    # Draw the time
    time_text = font.render("Time: " + str(frames // 60), True, (255, 255, 255))
    screen.blit(time_text, (0, 0))
    score_text = font.render("Score: " + str(score), True, (255, 255, 255))
    screen.blit(score_text, (0, 24))

    pygame.display.flip()
    # Wait for 1/60th of a second
    pygame.time.wait(1000 // 60)
    frames += 1

# Quit pygame
pygame.quit()

