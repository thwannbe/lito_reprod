# LITO Reproduction

This repository contains the reproduction of experiments from the paper titled:

**LANGUAGE-INTERFACED TABULAR OVERSAMPLING VIA PROGRESSIVE IMPUTATION AND SELF AUTHENTICATION**

## Overview

This project aims to replicate the experiments presented in the paper and validate the results.

## How to Run

1. Clone this repository:
```bash
git clone https://github.com/thwannbe/lito_reprod.git
```

2. Clone be_great repository:
```bash
git clone https://github.com/kathrinse/be_great.git
```

3. Copy lito.py to be_great directory:
```bash
cp lito_reprod/lito.py be_great/
```

4. Go to be_great directory and prepare TLM and dataset
```bash
cd be_great
# get TLM and dataset
```

5. Run lito.py with Python on the be_great directory:
```bash
python3 lito.py
```