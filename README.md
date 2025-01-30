# parkingspace

## Parking Slot Detection Model:
This is a model used to detect the available parking slots which was trained with yolov5 object detection model. It is trained in google colab using dataset from the roboflow.
> ⚠️ **WARNING:** This project refuses to run on Windows. Why?
 > Because Windows is **bullsh*t** haha! 💩
> Save your time **switch to Linux!** 🐧🚀

### Google colab:

Below is the link of the google colab link where the model was trained with datasets. And the best.pt was exported for integrating it with api.

[Google colab link](https://colab.research.google.com/drive/18Wz7rz7IeWxsfgTjwLiDtkHnNX-aKuCt?usp=sharing )

### Setting up for development:

1.  Clone the github repository and move into the directory

```
git clone --recurse-submodules -j8 https://github.com/Luxxgit2k4/parking.git
cd parking
```

3. Install python directly from the site or using commands with the distro you use: ( I use Arch btw). And then install pyenv:

```
sudo pacman -S python
sudo pacman -S pyenv
```

- pyenv can also be installed using the following command:

```
curl https://pyenv.run | bash
```


4. Add the following to your zsh or bash shell (`~/.bashrc`, `~/.zshrc`)

```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

- Then, restart your terminal or run:

```
source ~/.bashrc  or source ~/.zshrc
```

5. Install an Older Python Version using pyenv for this project as it uses an older version.

```
pyenv install 3.10.6
```

 - After installing the pyenv model run the following command inside the project directory to set a different python version for the project directory alone:

```
pyenv local 3.10.6
```

- Then continue with creating the virtual environment with the following command:

```
python -m venv environment_name
```

- Activate the environment:

```
source environment_name/bin/activate
```

- And check the python version inside the environment with

```
python --version
```

- Deactivate the environment using:

```
deactivate
```

