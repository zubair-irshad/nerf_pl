from progress.bar import Bar

def _in_ipynb():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    except ModuleNotFoundError:
        return False

class ETABar(Bar):
    """Progress bar that displays the estimated time of completion."""
    suffix = "%(percent).1f%% - %(eta)ds"
    bar_prefix = " "
    bar_suffix = " "
    empty_fill = "∙"
    fill = "█"

    def writeln(self, line: str):
        """Writes the line to the console.

        Description:
            This method is Jupyter notebook aware, and will do the
            right thing when in that environment as opposed to being
            run from the command line.

        Args:
            line (str): The message to write
        """
        if _in_ipynb():
            from IPython.display import clear_output
            clear_output(wait=True)
            self.fill = "#"
            print(line)
        else:
            Bar.writeln(self, line)

    def info(self, text: str):
        """Appends the given information to the progress bar message.

        Args:
            text (str): A status message for the progress bar.
        """
        self.suffix = "%(percent).1f%% - %(eta)ds {}".format(text)