import shutil
from rich.console import Console
from rich.text import Text

console = Console()

def get_terminal_width():
    return shutil.get_terminal_size().columns

def display_boolean_blocks(boolean_list, height=5, spacing=2):
    terminal_width = get_terminal_width()
    total_blocks = len(boolean_list)
    total_spacing = (total_blocks - 1) * spacing
    available_width = terminal_width - total_spacing
    width = min(4, max(1, available_width // total_blocks))

    for _ in range(height):
        text = Text()
        for i, boolean in enumerate(boolean_list):
            block = "█" * width
            if boolean:
                text.append(block, style="blue")
            else:
                text.append(block, style="red")
            if i < len(boolean_list) - 1:
                text.append(" " * spacing)
        console.print(text)

    number_text = Text()
    for i in range(len(boolean_list)):
        number = str(i + 1)
        centered_number = f"{number:^{width}}"
        number_text.append(centered_number)
        number_text.append(" " * spacing)
    console.print(number_text)

boolean_list = [True, False] * 20
display_boolean_blocks(boolean_list, height=5, spacing=2)