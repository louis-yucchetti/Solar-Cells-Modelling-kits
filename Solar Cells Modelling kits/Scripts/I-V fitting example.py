"""
==============================================================================
SOLAR CELL PARAMETER EXTRACTION USING LAMBERT W FUNCTION
==============================================================================
Based on:
- Montalvo-Galicia et al., Nanomaterials 2022 (MDPI)
- Single-diode model with Rs and Rsh

For Tandem Perovskite/Si-HIT Solar Cell
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.optimize import least_squares

# ==============================================================================
# 1. EXPERIMENTAL DATA (from SCAPS simulation)
# ==============================================================================
# Format: index, J (mA/cm²), V (V)
# J is negative in SCAPS convention, we convert to positive

raw_data = """
0	-17.49396	0.8494823
1	-17.31726	1.760559
2	-17.14055	1.798877
3	-16.96384	1.819213
4	-16.78714	1.833277
5	-16.61043	1.844728
6	-16.43372	1.85392
7	-16.25701	1.861593
8	-16.08031	1.868336
9	-15.9036	1.874261
10	-15.72689	1.879551
11	-15.55019	1.884324
12	-15.37348	1.888659
13	-15.19677	1.892623
14	-15.02007	1.896268
15	-14.84336	1.89963
16	-14.66665	1.902754
17	-14.48995	1.905662
18	-14.31324	1.908387
19	-14.13653	1.910944
20	-13.95983	1.913367
21	-13.78312	1.915663
22	-13.60641	1.917857
23	-13.42971	1.919964
24	-13.253	1.926477
25	-13.07629	1.929117
26	-12.89959	1.931757
27	-12.72288	1.934397
28	-12.54617	1.936706
29	-12.36947	1.938605
30	-12.19276	1.940505
31	-12.01605	1.942405
32	-11.83935	1.944266
33	-11.66264	1.945752
34	-11.48593	1.947238
35	-11.30923	1.948724
36	-11.13252	1.95021
37	-10.95581	1.951696
38	-10.77911	1.953183
39	-10.6024	1.954669
40	-10.42569	1.956155
41	-10.24899	1.957641
42	-10.07228	1.959127
43	-9.895574	1.960613
44	-9.718868	1.962099
45	-9.542161	1.963585
46	-9.365454	1.965071
47	-9.188748	1.966557
48	-9.012041	1.968043
49	-8.835334	1.969391
50	-8.658628	1.970479
60	-6.891561	1.981359
70	-5.124494	1.992061
80	-3.357427	2.001039
90	-1.590360	2.008953
99	0.0	2.015106
"""

# Parse data
lines = [l.strip() for l in raw_data.strip().split('\n') if l.strip()]
V_data = []
J_data = []
for line in lines:
    parts = line.split()
    J_data.append(-float(parts[1]))  # Convert to positive
    V_data.append(float(parts[2]))

V = np.array(V_data)
J = np.array(J_data)

# Add V=0 point
V = np.insert(V, 0, 0.0)
J = np.insert(J, 0, J[0])

# ==============================================================================
# 2. EXTRACT MEASURED PV PARAMETERS
# ==============================================================================
Jsc = J[0]          # Short-circuit current (mA/cm²)
Voc = V[-1]         # Open-circuit voltage (V)

# Power and MPP
P = V * J
idx_mpp = np.argmax(P)
Vmpp = V[idx_mpp]   # Voltage at MPP
Jmpp = J[idx_mpp]   # Current at MPP
Pmax = Vmpp * Jmpp  # Maximum power (mW/cm²)

# Fill Factor and Efficiency
FF = Pmax / (Voc * Jsc) * 100  # Fill Factor (%)
PCE = Pmax / 100 * 100          # PCE (%) assuming 100 mW/cm² input

print("="*70)
print("   MEASURED PV PARAMETERS")
print("="*70)
print(f"   Voc  = {Voc:.4f} V")
print(f"   Jsc  = {Jsc:.2f} mA/cm²")
print(f"   Vmpp = {Vmpp:.4f} V")
print(f"   Jmpp = {Jmpp:.2f} mA/cm²")
print(f"   Pmax = {Pmax:.2f} mW/cm²")
print(f"   FF   = {FF:.2f} %")
print(f"   PCE  = {PCE:.2f} %")
print("="*70)

# ==============================================================================
# 3. CONSTANTS
# ==============================================================================
T = 300                    # Temperature (K)
kB = 8.617333e-5           # Boltzmann constant (eV/K)
Vt = kB * T                # Thermal voltage (~0.0259 V at 300K)

# For tandem solar cell: effective Vt = 2*Vt (2 junctions in series)
Vt_eff = 2 * Vt

# ==============================================================================
# 4. SINGLE-DIODE MODEL WITH LAMBERT W FUNCTION
# ==============================================================================
def calc_J_lambert(V, n, Rs, Rsh, Jsc, Voc, Vt):
    """
    Calculate current using Lambert W function.
    This is the analytical solution of the single-diode equation:
    
    J = Jph - J0*(exp((V + J*Rs)/(n*Vt)) - 1) - (V + J*Rs)/Rsh
    
    Parameters:
    -----------
    V : array
        Voltage values (V)
    n : float
        Ideality factor
    Rs : float
        Series resistance (Ohm.cm²)
    Rsh : float
        Shunt resistance (Ohm.cm²)
    Jsc : float
        Short-circuit current (mA/cm²)
    Voc : float
        Open-circuit voltage (V)
    Vt : float
        Thermal voltage (V)
    
    Returns:
    --------
    J_calc : array
        Calculated current (mA/cm²)
    """
    nVt = n * Vt
    
    # Calculate J0 from boundary conditions
    J0 = (Jsc - (Voc - Rs*Jsc)/Rsh) / (np.exp(Voc/nVt) - 1)
    
    # Calculate Jph (photo-generated current)
    Jph = Jsc * (1 + Rs/Rsh) + J0 * (np.exp(Rs*Jsc/nVt) - 1)
    
    J_calc = np.zeros_like(V)
    
    for i, v in enumerate(V):
        try:
            # Argument for Lambert W function
            arg = (J0 * Rs / nVt) * np.exp((Rs * (Jph + J0) + v) / (nVt * (1 + Rs/Rsh)))
            
            # Lambert W function (principal branch)
            W = np.real(lambertw(arg))
            
            # Calculated current
            J_calc[i] = (Jph + J0 - v/Rsh) / (1 + Rs/Rsh) - (nVt/Rs) * W
            
        except:
            J_calc[i] = 0
    
    return np.maximum(J_calc, 0)

# ==============================================================================
# 5. RESIDUAL FUNCTION FOR LEAST SQUARES FIT
# ==============================================================================
def residuals(params, V, J_exp, Jsc, Voc, Vt):
    """
    Residual function for least squares optimization.
    
    Parameters:
    -----------
    params : array
        [n, Rs, Rsh] - parameters to fit
    V : array
        Voltage values
    J_exp : array
        Experimental current values
    Jsc, Voc : float
        Measured parameters
    Vt : float
        Thermal voltage
    
    Returns:
    --------
    residuals : array
        J_calc - J_exp
    """
    n, Rs, Rsh = params
    
    # Physical constraints
    if n < 1 or Rs <= 0 or Rsh <= 0:
        return np.ones_like(J_exp) * 1e10
    
    try:
        J_calc = calc_J_lambert(V, n, Rs, Rsh, Jsc, Voc, Vt)
        return J_calc - J_exp
    except:
        return np.ones_like(J_exp) * 1e10

# ==============================================================================
# 6. INITIAL PARAMETER ESTIMATION
# ==============================================================================
# From Villalva et al. (2009) and Montalvo-Galicia et al. (2022)

n0 = 1.3                           # Initial ideality factor
Rs0 = (Voc - Vmpp) / Jmpp          # Initial series resistance
Rsh0 = Vmpp / (Jsc - Jmpp)         # Initial shunt resistance

print("\n" + "="*70)
print("   INITIAL PARAMETER ESTIMATES")
print("="*70)
print(f"   n   = {n0:.4f}")
print(f"   Rs  = {Rs0:.4f} Ω·cm²")
print(f"   Rsh = {Rsh0:.2f} Ω·cm²")
print("="*70)

# ==============================================================================
# 7. PARAMETER FITTING WITH LEAST SQUARES
# ==============================================================================
print("\n   Fitting with least_squares (Lambert W method)...")

# Initial values and bounds
b0 = [n0, Rs0, Rsh0]
bounds = ([1.0, 0.0001, 1], [3.0, 10, 1e6])

# Perform fit
result = least_squares(
    residuals, 
    b0, 
    args=(V, J, Jsc, Voc, Vt_eff),
    bounds=bounds, 
    method='trf',      # Trust Region Reflective
    ftol=1e-12, 
    xtol=1e-12,
    verbose=0
)

# Extract fitted parameters
n_fit, Rs_fit, Rsh_fit = result.x

# Calculate derived parameters
nVt = n_fit * Vt_eff
J0_fit = (Jsc - (Voc - Rs_fit*Jsc)/Rsh_fit) / (np.exp(Voc/nVt) - 1)
Jph_fit = Jsc * (1 + Rs_fit/Rsh_fit) + J0_fit * (np.exp(Rs_fit*Jsc/nVt) - 1)

# Calculate fitted curve
J_fit = calc_J_lambert(V, n_fit, Rs_fit, Rsh_fit, Jsc, Voc, Vt_eff)

# ==============================================================================
# 8. FIT QUALITY METRICS
# ==============================================================================
# Root Mean Square Error
RMSE = np.sqrt(np.mean((J - J_fit)**2))

# Coefficient of determination (R²)
SS_res = np.sum((J - J_fit)**2)
SS_tot = np.sum((J - np.mean(J))**2)
R2 = 1 - SS_res / SS_tot

# Maximum power error
P_fit = V * J_fit
Pmax_fit = np.max(P_fit)
Pmax_error = abs(Pmax_fit - Pmax) / Pmax * 100

print("\n" + "="*70)
print("   FITTED PARAMETERS (LAMBERT W METHOD)")
print("="*70)
print(f"   Ideality factor        n    = {n_fit:.4f}")
print(f"   Series resistance      Rs   = {Rs_fit:.4f} Ω·cm²")
print(f"   Shunt resistance       Rsh  = {Rsh_fit:.2f} Ω·cm²")
print(f"   Saturation current     J0   = {J0_fit:.4e} mA/cm²")
print(f"   Photo-generated curr.  Jph  = {Jph_fit:.4f} mA/cm²")
print("="*70)

print("\n" + "="*70)
print("   FIT QUALITY")
print("="*70)
print(f"   RMSE          = {RMSE:.6f} mA/cm²")
print(f"   R²            = {R2:.6f}")
print(f"   Pmax error    = {Pmax_error:.4f} %")
print("="*70)

# ==============================================================================
# 9. PLOTTING
# ==============================================================================
fig = plt.figure(figsize=(14, 10))

# --- Plot 1: J-V Curve with Fit ---
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(V, J, 'ko', markersize=6, alpha=0.7, label='Experimental data')

# Smooth curve for fit
V_smooth = np.linspace(0, Voc * 1.01, 200)
J_smooth = calc_J_lambert(V_smooth, n_fit, Rs_fit, Rsh_fit, Jsc, Voc, Vt_eff)
ax1.plot(V_smooth, J_smooth, 'r-', linewidth=2.5, label=f'Lambert W fit (R²={R2:.4f})')

# Mark key points
ax1.scatter([0], [Jsc], color='blue', s=120, zorder=5, marker='s', 
            edgecolor='black', label=f'Jsc = {Jsc:.2f} mA/cm²')
ax1.scatter([Voc], [0], color='green', s=120, zorder=5, marker='s', 
            edgecolor='black', label=f'Voc = {Voc:.3f} V')
ax1.scatter([Vmpp], [Jmpp], color='purple', s=120, zorder=5, marker='s', 
            edgecolor='black', label=f'MPP ({Vmpp:.2f}V, {Jmpp:.1f}mA/cm²)')

ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.axvline(x=0, color='gray', linewidth=0.5)
ax1.set_xlabel('Voltage V (V)', fontsize=12)
ax1.set_ylabel('Current Density J (mA/cm²)', fontsize=12)
ax1.set_title('J-V Characteristic with Single-Diode Model Fit', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.05, 2.15)
ax1.set_ylim(-0.5, 20)

# --- Plot 2: Residuals ---
ax2 = fig.add_subplot(2, 2, 2)
residuals_plot = J - J_fit
ax2.bar(V, residuals_plot, width=0.015, color='steelblue', alpha=0.7, edgecolor='navy')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Voltage V (V)', fontsize=12)
ax2.set_ylabel('Residual (mA/cm²)', fontsize=12)
ax2.set_title('Fit Residuals (Data - Model)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.05, 2.15)

# --- Plot 3: P-V Curve ---
ax3 = fig.add_subplot(2, 2, 3)
P_smooth = V_smooth * J_smooth
ax3.plot(V, P, 'ko', markersize=6, alpha=0.7, label='Experimental data')
ax3.plot(V_smooth, P_smooth, 'r-', linewidth=2, label='Lambert W fit')
ax3.fill_between(V_smooth, 0, P_smooth, alpha=0.2, color='green')
ax3.scatter([Vmpp], [Pmax], color='purple', s=150, zorder=5, 
            edgecolor='black', linewidth=2, label=f'Pmax = {Pmax:.1f} mW/cm²')
ax3.axvline(x=Vmpp, color='purple', linestyle='--', alpha=0.5)
ax3.set_xlabel('Voltage V (V)', fontsize=12)
ax3.set_ylabel('Power Density P (mW/cm²)', fontsize=12)
ax3.set_title('Power vs Voltage', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.05, 2.15)
ax3.set_ylim(0, 35)

# --- Plot 4: Summary Table ---
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

summary_text = f"""
┌─────────────────────────────────────────────────────────┐
│     SINGLE-DIODE MODEL - PARAMETER EXTRACTION           │
│     Using Lambert W Function Method                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  MEASURED PARAMETERS:                                   │
│  ─────────────────────────────────────────────────────  │
│    Voc  = {Voc:8.4f} V      Jsc  = {Jsc:6.2f} mA/cm²      │
│    Vmpp = {Vmpp:8.4f} V      Jmpp = {Jmpp:6.2f} mA/cm²     │
│    Pmax = {Pmax:8.2f} mW/cm²  FF   = {FF:6.2f} %          │
│    PCE  = {PCE:8.2f} %                                   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  FITTED DIODE PARAMETERS:                               │
│  ─────────────────────────────────────────────────────  │
│    Ideality factor       n   = {n_fit:8.4f}              │
│    Series resistance     Rs  = {Rs_fit:8.4f} Ω·cm²       │
│    Shunt resistance      Rsh = {Rsh_fit:8.2f} Ω·cm²      │
│    Saturation current    J0  = {J0_fit:8.2e} mA/cm²      │
│    Photo-current         Jph = {Jph_fit:8.4f} mA/cm²     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  FIT QUALITY:                                           │
│  ─────────────────────────────────────────────────────  │
│    R²         = {R2:.6f}                                 │
│    RMSE       = {RMSE:.6f} mA/cm²                        │
│    Pmax error = {Pmax_error:.4f} %                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.5))

plt.tight_layout()
plt.savefig('Tandem_JV_LambertW_Fit.png', dpi=200, bbox_inches='tight')
print("\n   Figure saved: Tandem_JV_LambertW_Fit.png")

# ==============================================================================
# 10. SAVE DATA TO CSV
# ==============================================================================
output_data = np.column_stack([V, J, J_fit, J - J_fit, P])
header = 'V(V),J_exp(mA/cm2),J_fit(mA/cm2),Residual(mA/cm2),P(mW/cm2)'
np.savetxt('Tandem_JV_Fit_Data.csv', output_data, delimiter=',', 
           header=header, comments='', fmt='%.6f')
print("   Data saved: Tandem_JV_Fit_Data.csv")

# ==============================================================================
# 11. DISPLAY RESULTS SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("   FINAL SUMMARY")
print("="*70)
print(f"""
   Single-Diode Model: J = Jph - J0*(exp((V+J*Rs)/(n*Vt))-1) - (V+J*Rs)/Rsh
   
   With Lambert W analytical solution:
   
   ┌────────────────────────────────────────┐
   │  n   = {n_fit:.4f}   (ideality factor)   │
   │  Rs  = {Rs_fit:.4f} Ω·cm² (series R)     │
   │  Rsh = {Rsh_fit:.2e} Ω·cm² (shunt R)  │
   │  J0  = {J0_fit:.2e} mA/cm² (sat. curr)│
   │  Jph = {Jph_fit:.4f} mA/cm² (photo curr) │
   └────────────────────────────────────────┘
   
   Fit quality: R² = {R2:.6f}, RMSE = {RMSE:.4f} mA/cm²
""")
print("="*70)

plt.show()