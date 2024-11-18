# Shiny App Front End

Shiny is a Python library capable of displaying a locally hosted application which is able to update in real time to user inputs. This library was used to help visualise and document the development of the PH3010 group C coding project. Through this library we are able to clearly see how the changes to time scale, eccentricity of orbit and orbital axies can effect simulations of the orbit of Mercury around the Sun.

## Authors

Alex Boxer, Oliver Blease, Shreya Ghosh, Carolena Lukaszek, Thomas Mansfield, Harry Nsubuga.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Shiny.

Use pip to also install shinyswatch.

```bash
pip install shiny
pip install shinyswatch
```

## Usage

It is recommended to use Visual Studio Code of Pycharm to be able to load the application - Anaconda has some troubles loading the server.

Download this folder, all dependencies for the application are in this folder. Run ```app.py``` and select an option from the dropdown menu on the sidebar. 

Running the three body simulation may be computationally intensive and may require you to run ```final commits/3-Body Sun-Mercury-Venus.py``` on its own.

## License

[MIT](https://choosealicense.com/licenses/mit/)
