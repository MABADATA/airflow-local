class Envsetter:
  def __init__(self, requirements_file_path) -> None:
    """
     Parameters
        ----------
        requirements_file_path : str
            The path of the requirements.txt file
        """
    self.requirements = requirements_file_path

  def install_requirements(self):
    # Use the 'pip' command to install the packages listed in the requirements file
    try:
      os.system(f"pip install -r {self.requirements}")
      logging.info("Environment is set! requirements installed successfully!")
    except Exception as err:
      logging.error(f"Exception raised during requirements installation!\nException:\n{err}")