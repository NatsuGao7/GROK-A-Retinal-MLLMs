# Overview

This repository contains code and configuration for training with **RETFound** and **RetiZero** encoders.

## Folder Structure

- **`RETFound_MAE/`** – Contains the encoder implementation for **RETFound**.  
- **`retrieval/`** – Contains the encoder implementation for **RetiZero**.

## Training

To start training, simply run:

```bash
bash train_grok.sh
````

**Notes:**

* Modify the necessary file path variables inside the script to match your environment.
* The current training configuration uses `zero2.json`.

## Data Format

A sample data slice is provided in **`example_data.json`**.
Your generated **training** and **testing** datasets should follow the same structure as in this example.

```
```

