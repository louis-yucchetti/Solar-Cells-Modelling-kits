# @title 5. Simulation Robuste & Correction des Types (Fix TypeError)
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import numpy as np

# 1. Écriture du fichier SPICE
# Note : J'arrête le sweep à 0.8V car à 1.0V le courant explose (exponentielle)
spice_code = """
* Caractérisation PV Dark vs Light
.param I_sc = 0.035
.model D_PV D (Is=1e-9 N=1.5)

.subckt CELLULE_PV pos neg params: LIGHT=1
    I_gen neg int DC {I_sc * LIGHT}
    D1 int neg D_PV
    Rsh int neg 500
    Rs int pos 0.02
.ends

* Banc de test
V_sweep commun 0 DC 0

* Cellule Light
V_L commun n1 DC 0
X1 n1 0 CELLULE_PV params: LIGHT=1

* Cellule Dark
V_D commun n2 DC 0
X2 n2 0 CELLULE_PV params: LIGHT=0

.dc V_sweep -0.2 0.8 0.01
.print dc i(V_L) i(V_D)
.end
"""

with open("simulation_v3.cir", "w") as f:
    f.write(spice_code)

# 2. Exécution NGSPICE
os.system("ngspice -b simulation_v3.cir > simulation_output.txt")

# 3. Lecture brute du fichier
data_lines = []
capture = False

try:
    with open("simulation_output.txt", "r") as f:
        for line in f:
            if "Index" in line:
                capture = True
                continue
            if capture and "-----" in line: continue
            if capture and line.strip() == "": continue

            if capture:
                parts = line.split()
                # On ne garde que les lignes qui ressemblent à des données (au moins 4 colonnes)
                if len(parts) >= 4:
                    data_lines.append(f"{parts[1]} {parts[2]} {parts[3]}")

except FileNotFoundError:
    print("❌ Erreur : Fichier de sortie introuvable.")

# 4. Conversion et Nettoyage (LE FIX EST ICI)
if len(data_lines) > 0:
    # Création du DataFrame brut (tout est encore du texte ici)
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=" ", names=["V", "I_Light", "I_Dark"])

    # --- CONVERSION FORCÉE EN NOMBRES ---
    # 'errors=coerce' transforme les textes récalcitrants en NaN
    df["V"] = pd.to_numeric(df["V"], errors='coerce')
    df["I_Light"] = pd.to_numeric(df["I_Light"], errors='coerce')
    df["I_Dark"] = pd.to_numeric(df["I_Dark"], errors='coerce')

    # Suppression des lignes invalides (NaN)
    df = df.dropna()

    # Vérification
    print(f"✅ Données nettoyées : {len(df)} points valides.")
    print(df.head()) # Affiche les premières lignes pour vérifier

    # 5. Traçage
    plt.figure(figsize=(10, 6))

    # Maintenant que ce sont des floats, l'opérateur "-" fonctionne
    plt.plot(df["V"], -df["I_Light"] * 1000, 'r-', linewidth=2, label="Lumière (1 Sun)")
    plt.plot(df["V"], -df["I_Dark"] * 1000, 'k--', label="Dark (Obscurité)")

    plt.title("Caractéristique I-V (Simulation NGSPICE)")
    plt.xlabel("Tension (V)")
    plt.ylabel("Courant (mA)")
    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)

    # Limites pour voir la partie utile (Voc) sans être écrasé par le courant direct
    plt.ylim(-10, 40)
    plt.grid(True)
    plt.legend()
    plt.show()

else:
    print("⚠️ Aucune donnée récupérée dans le fichier texte.")
	
	