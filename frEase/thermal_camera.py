import shutil
from rich.console import Console
from rich.text import Text

class ThermalCamera:
    def __init__(self):
        self.console = Console()

    def get_terminal_width(self):
        return shutil.get_terminal_size().columns

    def display_network_state(self, boolean_list, height=5, spacing=2):
        terminal_width = self.get_terminal_width()
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
            self.console.print(text)

        number_text = Text()
        for i in range(len(boolean_list)):
            number = str(i + 1)
            centered_number = f"{number:^{width}}"
            number_text.append(centered_number)
            number_text.append(" " * spacing)
        self.console.print(number_text)