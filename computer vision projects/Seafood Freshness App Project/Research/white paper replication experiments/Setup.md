`Install uv package manager:`
1.	Open terminal/powershell
2.	Paste and run this command: curl -LsSf https://astral.sh/uv/install.sh | sh if MacOS/Linux
3.	Paste and run this command: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" if Windows OS
4.	Check installation : uv –version

`Create virtual env:`
1.	cd “sardine-mackerel white paper replication”
2.	uv python install 3.9
3.	uv venv --python 3.9 .venv
4.	source .venv/bin/activate for mac/linux 
	.venv\Scripts\activate for windows
5.	uv pip install -r requirements.txt
	normal pip install -r requirements.txt showed some conflicts between numpy and opencv for me in windows, so I used uv pip
6.  Install graphviz for model architecture plotting: 
	
	First check: dot -V
	If you get, 'dot' is not recognized...
	then Graphviz isn't installed or isn't on your PATH.

	Mac - sudo port install graphviz
	Linux - 1. sudo apt update 2.sudo apt install graphviz
	Windows - https://graphviz.org/download/

	After installation, ensure this folder exists:
	C:\Program Files\Graphviz\bin

	Then add it to your PATH if the installer didn't.
	Open a new Command Prompt and run: dot -V
	You should see something like: dot - graphviz version 12.2.1
	Close the Command Prompt and Run your script again.
7. Check if opencv is installed properly:
	python
	import cv2
	exit()
	if it throws any errors, run the below commands in linux terminal
	sudo apt update
	sudo apt install -y libgl1

If that shows an error, refer to these commands
sudo apt-get install -y libgl1 libglib2.0-0