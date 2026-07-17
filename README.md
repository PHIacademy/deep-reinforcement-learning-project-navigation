# Project: Navigation — Deep Reinforcement Learning Nanodegree

### Introduction

For this project, an agent is trained to navigate (and collect bananas!) in a large, square world using a Deep Q-Network (DQN).

| Random Agent | Trained Agent |
|---|---|
| ![Random Agent](Training-Results/random_agent.gif) | ![Trained Agent](Training-Results/trained_agent.gif) |

A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

**State space**: 37 dimensions, containing the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

**Action space**: 4 discrete actions:

- `0` — move forward
- `1` — move backward
- `2` — turn left
- `3` — turn right

**Solve criteria**: The task is episodic. The environment is considered solved when the agent achieves an average score of **+13 over 100 consecutive episodes**.

---

### Getting Started

#### 1. Install Python 3.6 in a virtual environment

The Unity ML-Agents API used in this project (`unityagents`, ml-agents 0.4) requires Python 3.6.

```bash
winget install -e --id Python.Python.3.6 --accept-source-agreements --accept-package-agreements --override "InstallAllUsers=0 PrependPath=0 Include_doc=0 Include_test=0 Include_tcltk=0 Include_tools=0 Include_launcher=0 AssociateFiles=0 Shortcuts=0 CompileAll=0 Include_pip=1 SimpleInstall=1"
```

Create and activate the project virtual environment:

```bash
cd /d/python/DQN_project
py -3.6 -m venv .venv
source .venv/Scripts/activate
python --version   # should print Python 3.6.8
```

Pin installer tooling for 3.6 compatibility:

```bash
python -m pip install --upgrade "pip<22" "setuptools<60" wheel
```

#### 2. Get the project code and the `unityagents` package

Clone the Udacity `Value-based-methods` repository and install the `unityagents` package it ships with (this is what provides `from unityagents import UnityEnvironment`):

```bash
cd /d/python/DQN_project
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
pip install .
```

#### 3. Install the remaining dependencies

```bash
pip install "torch==1.10.2" "matplotlib<3.4" "ipykernel<6"
```

Quick validation:

```bash
python -c "import torch, matplotlib, unityagents; print(torch.__version__, matplotlib.__version__)"
```

#### 4. Download the Unity environment

Download the pre-built Banana environment matching your OS (you do **not** need to install Unity itself):

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Unzip it into `Value-based-methods/p1_navigation/` (or any folder of your choice), then point the `file_name` argument in `Navigation.ipynb` at the extracted `.exe` / `.app` / `.x86_64`. For example, on Windows (64-bit), if you unzipped the environment into the `p1_navigation` folder alongside the notebook:

```python
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
```

On Mac:

```python
env = UnityEnvironment(file_name="Banana.app")
```

Adjust the path to match wherever you actually placed the downloaded environment on your machine.

#### 5. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

> In VS Code, the kernel may show as `.venv (3.6.8)` instead of `drlnd` — both point to the same environment and are fine to use.

---

### File Structure

| File | Description |
|---|---|
| `Navigation.ipynb` | Main notebook — launches the environment, trains the agent, and watches the trained agent play |
| `dqn_agent.py` | The `Agent` class — epsilon-greedy action selection, experience replay, and the fixed-Q-targets DQN learning update |
| `model.py` | `QNetwork` — the PyTorch network used as the function approximator |
| `checkpoint.pth` | Saved weights of the trained agent (produced by running section 4 of the notebook) |
| `Training-Results/random_agent.gif` | Recording of an untrained agent selecting actions randomly |
| `Training-Results/trained_agent.gif` | Recording of the trained agent navigating and collecting bananas |

---

### Instructions

Open `Navigation.ipynb` and run the cells **in order**. The notebook is split into two flows depending on what you want to do:

#### Flow A — Train the agent from scratch

Run:

1. **Section 1** — Start the Environment (launches Unity; `no_graphics=True` is recommended here for faster training)
2. The "get the default brain" cell
3. **Section 2** — Examine the State and Action Spaces
4. **Section 4** — creates the `Agent` and runs the `dqn()` training loop

Training runs until either the environment is solved (average score ≥ +13 over 100 consecutive episodes) or 2000 episodes are reached. On success, the trained weights are automatically saved to `checkpoint.pth`, and a plot of score-per-episode is displayed. `env.close()` is called automatically at the end of this cell.

> Section 3 (random-action demo) is optional — it's only there to sanity-check that the environment is wired up correctly before training. You can skip straight from Section 2 to Section 4.

#### Flow B — Watch a previously-trained agent (no training)

If you already have a `checkpoint.pth` from a prior training run (or want to watch the agent right after training, in the same notebook run):

1. **Restart the kernel** if training already ran and closed the environment in this session — reopening a Unity environment in the same Windows Python process after `env.close()` is unreliable with this version of `unityagents`.
2. Run the imports cell, then **Section 1** with `no_graphics=False` so you can see the window.
3. Run the "get the default brain" cell and **Section 2** (needed to reconstruct `state_size`/`action_size`).
4. **Section 5** — loads `checkpoint.pth` into a fresh `Agent` and runs one episode with greedy (non-random) action selection, so you can watch the trained agent play.

```python
agent = Agent(state_size=state_size, action_size=action_size, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
```

5. `env.close()` at the end of the section shuts down the Unity window when you're done.

---

### Results

The agent solved the environment (average score ≥ +13 over 100 consecutive episodes) in **388 episodes**. See the score plot at the end of Section 4 in `Navigation.ipynb`.

Running the trained agent greedily for one episode (Section 5) achieved a score of **17.0**, well above the +13 solve threshold.

---

### Notes

- See [`Report.md`](Report.md) for a full description of the learning algorithm, hyperparameters, network architecture, rewards plot, and ideas for future work.
- If Unity fails to launch with `Error: Global Illumination requires a graphics device to render albedo.`, this is expected behavior of `no_graphics=True` and can be ignored; it only affects rendering, not training.
