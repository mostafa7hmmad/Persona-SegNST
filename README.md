# Project Pipeline — Segmentation + Neural Style Transfer
![img](styled_image(6).png)

## Folder Structure
```
project/
│── main.py            # Main entry point
│── runner.py          # Auto-run script
│── config.py          # Configuration constants
│── patch_keras.py     # Keras patch for segmentation_models
│── models/
│   ├── __init__.py
│   ├── transformer.py # Style transfer model definition
│   └── segmentation.py# Segmentation model loader
│── utils/
│   ├── __init__.py
│   ├── stylize.py     # Stylizing helper function
│   └── keyboard.py    # Virtual keyboard UI logic

---

## Pipeline Flow

1. **runner.py**
    - Calls `run_app()` in `main.py`.

2. **main.py (run_app function)**
    - Loads pre-trained style models (.pth files) into memory.
    - Loads the segmentation model (.h5 file).
    - Opens webcam feed and processes each frame.
    - Passes frame to segmentation model → obtains mask.
    - Applies style transfer depending on user selection.
    - Updates virtual keyboard UI.
    - Displays the result.
    - Handles user mouse events for keyboard selection.

3. **models/transformer.py**
    - Contains the `TransformerNet` class (Neural Style Transfer).

4. **models/segmentation.py**
    - Loads the segmentation model (`best_Unet.h5`).
    - Contains the `predict()` function for mask generation.

5. **utils/stylize.py**
    - Contains `stylize()` function that applies style transfer to a frame.

6. **utils/keyboard.py**
    - Handles drawing and interacting with the virtual keyboard UI.

7. **config.py**
    - Defines constants, paths, preprocessing functions, style models list.
    - Ensures the keras patch is loaded before segmentation_models.

8. **patch_keras.py**
    - Fixes `keras.utils.generic_utils` bug for compatibility.

---

## User Flow

- Run:
    ```bash
    python runner.py
    ```
- Webcam feed opens with a virtual keyboard.
- User selects:
    - Style transfer mode (`Candy`, `Mosaic`, etc.)
    - Segmentation mode
    - Style on mask or background
- Press `q` to quit.

