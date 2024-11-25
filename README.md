# **Multi-Objective Optimization for Asset Allocation**

This project implements a multi-objective optimization algorithm based on **NSGA-II** to solve an asset allocation problem. The goal is to maximize the expected return while minimizing risk (variance) under certain constraints.

---

## **Table of Contents**
1. [Description](#description)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Results](#results)
7. [Authors](#authors)

---

## **Description**

This project uses Python libraries to:
- Load a dataset of financial assets from a CSV file.
- Apply a multi-objective evolutionary algorithm (NSGA-II) to optimize asset allocations.
- Generate a CSV file containing the optimal solutions (Pareto front).
- Visualize the Pareto front graphically.

The constraints include limits on the proportions allocated to asset classes such as:
- **Cash**
- **Precious Metals**
- **Bonds**
- **Stocks**

---

## **Prerequisites**

Before running this project, make sure you have installed the following tools and libraries:

- Python (>= 3.8)
- Python Libraries:
  - `numpy`
  - `pandas`
  - `deap`
  - `matplotlib`

---

## **Installation**

1. **Clone the project:**
   ```bash
   git clone https://github.com/your-repository/optimization-portfolio.git
   cd optimization-portfolio
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate    # On Linux/MacOS
   env\Scripts\activate       # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

1. **Prepare the data file:**
   - Ensure that the `assets_data.csv` file is in the main project directory.
   - The file should contain the following columns:
     - `mu`: Expected return.
     - `sigma2`: Variance (risk).
     - `class`: Asset class (e.g., cash, metals, bonds, stocks).

2. **Run the main script:**
   ```bash
   python main.py
   ```

3. **Results:**
   - A `pareto_results.csv` file is generated with the optimal solutions.
   - A figure `pareto_front.png` is saved, illustrating the Pareto front.

---

## **File Structure**

```plaintext
📁 optimization-portfolio/
├── main.py                # Main script
├── assets_data.csv        # Financial assets data file
├── pareto_results.csv     # Optimal portfolio results (generated after execution)
├── pareto_front.png       # Pareto front graph (generated after execution)
├── requirements.txt       # List of Python dependencies
├── README.md              # Project documentation
```

---

## **Results**

### **Example of Optimal Solutions:**
After execution, the optimal solutions are available in `pareto_results.csv`. Each row contains:
- The weights of the assets in the portfolio.
- The expected return.
- The risk (variance).

### **Pareto Front Graph:**
The `pareto_front.png` file shows the graph of return versus risk.

---

## **Authors**

- **Hamza HAMDAOUI**  
  Student in **MIAGE AI2**.  
  Contact: [LinkedIn](https://www.linkedin.com/in/hamdaouihamza/)

---
