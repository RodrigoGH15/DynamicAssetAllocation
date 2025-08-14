# DynamicAssetAllocation

Dynamic system for asset allocation and hedging based on market regimes.



In case of recreating the process, the uv package manager is needed.

The simplest steps to install uv are (or the ones that work for me):



-Open cmd and type the following commands:



py -m pip install --upgrade pip



py -m pip install --user pipx



Now close and reopen cmd



type pipx ensurepath



if you get no se reconoció el comando pipx or similar then type:



setx "PATH" "%PATH;\[root to Python installation]"

pay attention to the " " quotation marks

generally the root is smth along the lines of:

C:\\Users\\\[username]\\AppData\\Roaming\\Python\\Python31\[X]\\Scripts



then close and reopen cmd again



finally type:


pipx install uv



pipx upgrade uv



Now you can happily close your console and open your favorite IDE (hopefully VS Code)



Here clone the repo and then type:

uv sync

All dependencies Will be cloned into the newly created virtual environment. Even the right Python versión Will be installed if needed, because uv is that cool.



With all of these steps, you will have the best package manager in the world and Will be able to safely run my amazing script

-RGH

