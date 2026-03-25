import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows

def calculate_array_factor(theta, N, d_lambda, A, phi):
    k = 2 * np.pi 
    AF = np.zeros_like(theta, dtype=complex)
    for n in range(N):
        AF += A[n] * np.exp(1j * phi[n]) * np.exp(1j * n * k * d_lambda * np.cos(theta))
    return np.abs(AF)

def analyze_pattern(theta, af_mag):
    af_norm = af_mag / np.max(af_mag)
    af_db = 20 * np.log10(af_norm + 1e-12) 
    
    peaks, _ = find_peaks(af_norm)
    if len(peaks) > 0:
        peak_values_db = af_db[peaks]
        sidelobes = peak_values_db[peak_values_db < -0.1] # Filter out main beams
        if len(sidelobes) > 0:
            sll = np.max(sidelobes)
            # If the highest sidelobe is basically negative infinity, it doesn't exist
            if sll < -100: 
                sll = None
        else:
            sll = None
    else:
        sll = None

    # --- HPBW LOGIC ---
    main_peak_idx = np.argmax(af_norm)
    left_idx = main_peak_idx
    while left_idx > 0 and af_db[left_idx] > -3.0:
        left_idx -= 1
        
    right_idx = main_peak_idx
    while right_idx < len(af_db) - 1 and af_db[right_idx] > -3.0:
        right_idx += 1
        
    hpbw_deg = np.degrees(abs(theta[right_idx] - theta[left_idx]))
    return af_db, hpbw_deg, sll

class AntennaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Antenna Array Synthesizer")
        self.root.geometry("500x380") # Made window slightly wider
        self.root.configure(padx=20, pady=20)

        # --- Input Fields ---
        tk.Label(root, text="Number of Elements (N):", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.entry_n = tk.Entry(root, width=10)
        self.entry_n.grid(row=0, column=1, sticky="w")
        self.entry_n.insert(0, "5") 

        tk.Label(root, text="Spacing (d in wavelengths):", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.entry_d = tk.Entry(root, width=10)
        self.entry_d.grid(row=1, column=1, sticky="w")
        self.entry_d.insert(0, "0.5")

        tk.Label(root, text="Magnitudes (comma-separated):").grid(row=2, column=0, sticky="w", pady=5)
        self.entry_mag = tk.Entry(root, width=40)
        self.entry_mag.grid(row=2, column=1, sticky="w")
        self.entry_mag.insert(0, "1, 1, 1, 1, 1") 

        tk.Label(root, text="Phases in Degrees (comma-separated):").grid(row=3, column=0, sticky="w", pady=5)
        self.entry_phase = tk.Entry(root, width=40)
        self.entry_phase.grid(row=3, column=1, sticky="w")
        self.entry_phase.insert(0, "0, 0, 0, 0, 0")

        # --- Buttons ---
        btn_frame = tk.Frame(root)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=15)
        
        # Main Action Button
        tk.Button(btn_frame, text="Calculate & Plot", command=self.run_simulation, bg="lightblue", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Preset Loaders
        tk.Button(btn_frame, text="Load Uniform", command=self.load_uniform).grid(row=1, column=0, padx=5)
        tk.Button(btn_frame, text="Load Binomial", command=self.load_binomial).grid(row=1, column=1, padx=5)
        tk.Button(btn_frame, text="Load Chebyshev (-30dB)", command=self.load_chebyshev).grid(row=1, column=2, padx=5)

        # --- Output Labels ---
        self.lbl_hpbw = tk.Label(root, text="HPBW: --", font=("Arial", 12))
        self.lbl_hpbw.grid(row=5, column=0, columnspan=2, pady=2)
        
        self.lbl_sll = tk.Label(root, text="SLL: --", font=("Arial", 12))
        self.lbl_sll.grid(row=6, column=0, columnspan=2)

    # --- Preset Helper Functions ---
    def load_uniform(self):
        try:
            N = int(self.entry_n.get())
            self.entry_mag.delete(0, tk.END)
            self.entry_mag.insert(0, ", ".join(["1"] * N))
            self.entry_phase.delete(0, tk.END)
            self.entry_phase.insert(0, ", ".join(["0"] * N))
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for N first.")

    def load_binomial(self):
        self.entry_n.delete(0, tk.END)
        self.entry_n.insert(0, "5")
        self.entry_mag.delete(0, tk.END)
        self.entry_mag.insert(0, "1, 4, 6, 4, 1") 
        self.entry_phase.delete(0, tk.END)
        self.entry_phase.insert(0, "0, 0, 0, 0, 0")

    def load_chebyshev(self):
        try:
            N = int(self.entry_n.get())
            if N < 2:
                messagebox.showerror("Error", "N must be at least 2.")
                return
            
            # Generate Dolph-Chebyshev window targeted for -30 dB sidelobes
            w = windows.chebwin(N, at=30)
            w = w / np.max(w) # Normalize so the center element is 1
            
            # Format to 3 decimal places so it fits nicely in the text box
            mags = ", ".join([f"{x:.3f}" for x in w])
            
            self.entry_mag.delete(0, tk.END)
            self.entry_mag.insert(0, mags)
            self.entry_phase.delete(0, tk.END)
            self.entry_phase.insert(0, ", ".join(["0"] * N))
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for N first.")

    # --- Main Simulation ---
    def run_simulation(self):
        try:
            N = int(self.entry_n.get())
            d_lambda = float(self.entry_d.get())
            
            A = np.array([float(x.strip()) for x in self.entry_mag.get().split(',')])
            phi_deg = np.array([float(x.strip()) for x in self.entry_phase.get().split(',')])
            phi_rad = np.radians(phi_deg) 
            
            if len(A) != N or len(phi_rad) != N:
                messagebox.showerror("Input Error", f"N={N}, but you provided {len(A)} magnitudes and {len(phi_rad)} phases. They must match.")
                return

            theta = np.linspace(0, 2*np.pi, 36000) 
            af_mag = calculate_array_factor(theta, N, d_lambda, A, phi_rad)
            af_db, hpbw, sll = analyze_pattern(theta, af_mag)

            self.lbl_hpbw.config(text=f"HPBW: {hpbw:.2f}°")
            if sll is not None:
                self.lbl_sll.config(text=f"SLL: {sll:.2f} dB")
            else:
                self.lbl_sll.config(text="SLL: No sidelobes found!")

            plt.close('all') 
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
            ax.plot(theta, np.clip(af_db, -40, 0), color='blue', linewidth=1.5)
            
            # Optional: Draw a red dashed line indicating the SLL limit
            if sll is not None:
                ax.plot(theta, np.full_like(theta, sll), color='red', linestyle='--', linewidth=0.8, alpha=0.7)

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_rticks([-10, -20, -30, -40])
            ax.set_rmin(-40) 
            ax.set_rmax(0)
            ax.set_title(f"Array Factor (N={N}, d={d_lambda}λ)", va='bottom')
            plt.tight_layout()
            plt.show()

        except ValueError:
            messagebox.showerror("Data Error", "Please ensure inputs are numbers separated by commas.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AntennaApp(root)
    root.mainloop()