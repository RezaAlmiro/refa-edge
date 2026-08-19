# Beginner setup guide

This guide assumes you have never trained a model locally. Nothing here sends your
training data to a cloud service.

## 1. What you need

- A 64-bit Windows or Linux computer
- 32 GB system RAM is comfortable; 16 GB can run the smoke profile
- Python 3.10 or 3.11
- Git
- Around 5 GB free disk space
- Optional: an NVIDIA GPU with a current driver

An RTX 2060 commonly has 6 GB of VRAM. VRAM is the GPU's short-term working space;
it is different from your laptop's 32 GB of ordinary RAM.

## 2. Download the project on Windows

Open PowerShell in the folder where you want the project, then run each line:

~~~powershell
git clone https://github.com/RezaAlmiro/refa-edge.git
cd refa-edge
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -e .
~~~

The <code>.venv</code> folder is an isolated Python workspace. If anything goes
wrong, deleting only that folder and repeating these commands gives you a clean
software environment without touching the project or your other Python programs.

## 3. Check the computer

~~~powershell
.venv/Scripts/refa-edge.exe doctor
~~~

Expected GPU output resembles:

~~~text
REFA Edge hardware check
Python:       3.11.x
PyTorch:      2.x.x
CUDA visible: True
GPU:          NVIDIA GeForce RTX 2060
GPU memory:   6.0 GiB
Start with:   configs/rtx2060_6gb.yaml
~~~

If <code>CUDA visible</code> is false, the code still runs on the CPU. To use the
GPU, open the official [PyTorch selector](https://pytorch.org/get-started/locally/),
choose your operating system, Pip, Python, and the CUDA option recommended there.
Run its generated command with <code>.venv/Scripts/python.exe -m pip</code> in place
of plain <code>pip</code>. PyTorch packaging changes, so the live official command is
safer than a frozen command copied into this guide.

## 4. Prove the whole pipeline works

~~~powershell
.venv/Scripts/refa-edge.exe smoke
~~~

This creates a tiny synthetic dataset, trains four models, evaluates unseen samples,
and saves the measurements. A smoke run is a plumbing test; its accuracy is not a
scientific result.

Open <code>runs/smoke/results.csv</code> in a spreadsheet. The important columns are:

| Column | Plain meaning |
|---|---|
| <code>validation_accuracy</code> | Fraction of unseen questions answered correctly |
| <code>accuracy_supported</code> | Correctness when matching support exists |
| <code>accuracy_opposed</code> | Correctness when matching opposition exists |
| <code>accuracy_unknown</code> | Correctness when no exact evidence exists |
| <code>parameters</code> | Number of learned numbers in the model |
| <code>train_seconds</code> | Time for the configured training run |
| <code>examples_per_second</code> | Training speed |
| <code>peak_cuda_memory_mib</code> | Highest measured GPU memory allocation |

Random guessing is about 0.333 on this balanced three-answer task.

## 5. Run the laptop-sized experiment

First ask the program to try one genuine training batch:

~~~powershell
.venv/Scripts/refa-edge.exe fit-check --config configs/rtx2060_6gb.yaml --model refa
~~~

If the JSON says <code>"passed": true</code>, start the full comparison:

~~~powershell
.venv/Scripts/refa-edge.exe benchmark --config configs/rtx2060_6gb.yaml --models refa,gru,transformer,fast_stream
~~~

This is deliberately larger than the smoke run and can take time. Keep the laptop
plugged in, use its performance power mode, and make sure its air vents are clear.
Normal thermal protection should remain enabled.

If you receive a CUDA out-of-memory error:

1. Close GPU-heavy applications.
2. Open <code>configs/rtx2060_6gb.yaml</code> in a text editor.
3. Change <code>batch_size: 16</code> to <code>batch_size: 8</code>.
4. Save it and rerun the same command.
5. Record this change when sharing results.

If it still fails, change <code>d_model: 192</code> to 128,
<code>memory_rank: 48</code> to 32, and <code>num_heads: 6</code> to 4.

## 6. Ask the trained model through a browser

~~~powershell
.venv/Scripts/python.exe -m pip install -e ".[api]"
.venv/Scripts/refa-edge.exe serve --checkpoint runs/rtx2060/refa.pt
~~~

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), expand
<code>POST /predict</code>, choose “Try it out,” and use:

~~~json
{
  "evidence": [
    {"subject": 1, "relation": 2, "object": 3, "stance": "oppose"},
    {"subject": 1, "relation": 2, "object": 3, "stance": "support"}
  ],
  "query": {"subject": 1, "relation": 2, "object": 3}
}
~~~

The IDs do not have built-in human meanings in the synthetic task. They are symbols
used to test binding, memory, and later revision. The example asks whether relation 2
connects entity 1 to entity 3 after newer support revises older opposition.

Press Ctrl+C in PowerShell to stop the server.

## 7. Update later without losing results

~~~powershell
git pull
.venv/Scripts/python.exe -m pip install -e .
~~~

The <code>runs</code> folder is ignored by Git, so local result files are not
accidentally published. Copy important result folders somewhere backed up.
