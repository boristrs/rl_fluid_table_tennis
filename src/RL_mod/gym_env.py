"""
RL_mod.gym_env provides a Gym environment wrapper for the Plasma Pong game using Selenium to interact with the browser-based game.
"""

import io
import base64
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
from typing import Tuple, Dict 

class PlasmaPongEnv(gym.Env):
    """A Gymnasium environment for the Plasma Pong game using Selenium WebDriver.
    
    This environment wraps a browser-based Plasma Pong game and provides a standard
    interface for reinforcement learning agents to interact through keyboard actions.
    
    Attributes:
        render_mode (str | None): The rendering mode ("rgb_array" or "human").
        h (int): Height of the observation image (96 pixels).
        w (int): Width of the observation image (96 pixels).
        c (int): Color channels (3 for RGB).
        observation_space (Box): 96x96x3 RGB image space.
        action_space (Discrete): 5 discrete actions (0: none, 1: up, 2: down, 3: push, 4: suck).
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int=10000
    ) -> None:
        """Initialize the Plasma Pong environment.
        
        Args:
            render_mode: The rendering mode ("rgb_array" or "human"). Defaults to None.
            max_steps: Maximum steps per episode. Defaults to 10000.
        """
        super().__init__()
        self.render_mode = render_mode
        self.h, self.w, self.c = 96, 96, 3

        # Observation space: 96x96x3 RGB pixels (downsampled from 96x96 canvas)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(96, 96, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(5)  ## up, down, eject, suction, none

        # ?Key mappings (based on typical Pong controls: W=up, S=down, A=suck, D=push)
        self.action_keys = {
            0: [],
            1: [87],  # up
            2: [83],  # down
            3: [68],  # push plasma
            4: [65],  # suck plasma
        }
        # self.action_keys = {
        #     0: [],
        #     1: ["w"],  # up
        #     2: ["s"],  # down
        #     3: ["d"],  # push plasma
        #     4: ["a"],  # suck plasma
        # }

        # Set up headless Chrome
        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=448,700")  
        # chrome_options.add_argument("--window-size=96,96")  # Match canvas size

        # Path to chromedriver
        self.driver = webdriver.Chrome(options=chrome_options)

        # Path to index.html
        # self.html_path = os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), "..", "index.html")
        # )
        
        # via HTTP
        # Start the local server from the src with python -m http.server 8000
        self.html_path = "http://localhost:8000/index.html"

        # Previous lives for reward calculations
        self.prev_bot_life = 5
        self.prev_player_life = 5 # player (RL AI-agent)

        # Start the game
        self.reset()


    def reset(
        self,
        seed: int | None = None,
        options: dict | None =None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility. Defaults to None.
            options: Additional reset options. Defaults to None.
            
        Returns:
            A tuple containing:
                - observation: The initial game state as a 96x96x3 RGB image.
                - info: Additional information (empty dict).
        """
        # Load the game
        self.driver.get(self.html_path)
        time.sleep(2)  # Wait for the page to load

        # Ensure single-player mode (AI vs human)
        self.driver.execute_script("pong.ai.multiplayer = false;")

        # Reset game state
        self.driver.execute_script("restart();")
        
        # Wait for the game to display
        while not self.driver.execute_script("return pong.display;"):
            time.sleep(0.1)

        # Reset previous lives
        self.prev_bot_life = 5
        self.prev_player_life = 5

        # Focus on canvas by clicking it
        # To ensure the keys send are effective in the game
        self.driver.find_element(value="canvas").click()
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(
        self,
        action: int
        ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            action: An integer from the action space (0-4).
            
        Returns:
            A tuple containing:
                - observation: The current game state as a 96x96x3 RGB image.
                - reward: +1 for scoring, -1 for conceding, 0 otherwise.
                - terminated: True if the game is over.
                - truncated: False (not used).
                - info: Additional information (empty dict).
        """
        # Send action keys
        keys_to_send = self.action_keys[action]
        # print(f"Keys to send: {keys_to_send}")
            
        # Convert keycodes to key characters
        # keycode_to_char = {
        #     87: Keys.UP,      # W
        #     83: Keys.DOWN,    # S
        #     68: Keys.RIGHT,   # D
        #     65: Keys.LEFT,    # A
        # }
        keycode_to_char = {
            87: 'w',      # W
            83: 's',    # S
            68: 'd',   # D
            65: 'a',    # A
        }
        
        self.driver.find_element(value="canvas")
        # canvas.click()
        
        # Press down keys
        for keyCode in keys_to_send:
            if keyCode in keycode_to_char:
                char = keycode_to_char[keyCode]
                self.driver.execute_script(f"""
                var event = new KeyboardEvent('keydown', {{
                    key: '{char}',
                    code: 'Key{char.upper()}',
                    keyCode: {keyCode},
                    which: {keyCode},
                    bubbles: true,
                    cancelable: true
                }});
                window.dispatchEvent(event);
            """)
                
        # # Wait for frame to update (if 60 FPS, ~16ms per frame)
        time.sleep(0.016)
        
        # Release keys
        for keyCode in keys_to_send:
            if keyCode in keycode_to_char:
                char = keycode_to_char[keyCode]
                self.driver.execute_script(f"""
                    var event = new KeyboardEvent('keyup', {{
                        key: '{char}',
                        code: 'Key{char.upper()}',
                        keyCode: {keyCode},
                        which: {keyCode},
                        bubbles: true,
                        cancelable: true
                    }});
                    window.dispatchEvent(event);
                """)
        
        # for keyCode in keys_to_send:
        #     if keyCode in keycode_to_char:
        #         canvas.send_keys(keycode_to_char[keyCode])
        
        # Get Observation
        obs = self._get_obs()

        # Get rewards and done
        reward, done = self._get_reward_done()

        # Info (empty for now)
        info = {}
        truncated = False

        return obs, reward, done, truncated, info

    def _get_obs(
        self
        ) -> np.ndarray:
        """Capture the current game canvas as an observation.
        
        Returns:
            A numpy array of shape (96, 96, 3) representing the RGB image of the game canvas.
        """
        # Extract canvas via JS
        canvas_data_url = self.driver.execute_script(
            """
            var canvas = document.getElementById('canvas');
            return canvas.toDataURL('image/png');
        """
        )

        # Decode base64 to image
        encoded = canvas_data_url.split(",", 1)[1]
        data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(data)).convert("RGB")

        # Convert to numpy array (96.96.3)
        img_array = np.array(image)
        # Resize to 96.96.3 (96, 96, 3) RGB with the 3 ?
        # resized = cv2.resize(img_array, (96, 96)) # by default interpolation=cv2.INTER_LINEAR

        return img_array

    def _get_reward_done(
        self
        ) -> Tuple[float, bool]:
        """Calculate reward and check if the episode is done.
        
        Returns:
            A tuple containing:
                - reward: The reward value (1, -1, or 0).
                - done: True if the game is over (display inactive), False otherwise.
        """
        # Get current lives
        bot_life = self.driver.execute_script("return pong.ai.life;")
        player_life = self.driver.execute_script("return pong.player.life;")
        display_active = self.driver.execute_script("return pong.display;")
        # Reward:
        # +1 for player (RL AI-agent) scoring,
        # -1 for player conceding
        reward = 0
        if bot_life < self.prev_bot_life:
            reward = 1  # player (RL AI-agent) scored
        elif player_life < self.prev_player_life:
            reward = -1  # player (RL AI-agent) conceded

        # Update previous lives
        self.prev_bot_life = bot_life
        self.prev_player_life = player_life

        # Done: When display turn inactive (i.e., game over)
        done = not display_active

        return reward, done

    def render(
        self,
        mode: str ="human"
        ) -> np.ndarray | None:
        """Render the environment.
        
        Args:
            mode: The rendering mode ("rgb_array" or "human"). Defaults to "human".
            
        Returns:
            The RGB array if mode is "rgb_array", None for "human" mode.
            
        Raises:
            NotImplementedError: If an unsupported render mode is provided.
        """
        # Headless-friendly rendering
        if mode == "rgb_array":
            # Reuse your observation capture (e.g., JS canvas -> bytes -> np.array)
            # obs = self._eval("() => window._env_obs()")  # returns flattened uint8
            obs = self._get_obs()
            frame = np.array(obs, dtype=np.uint8).reshape(self.h, self.w, 3)
            return frame
        elif mode == "human":
           return None # Gymnasium conventions for environments where rendering is handled externally
        else:
            raise NotImplementedError(f"Unsupported render mode: {mode}")

    def close(self):
        self.driver.quit()
